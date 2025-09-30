"""Interactive Mini-o3 GUI demo.

This module provides a Gradio application for chatting with Mini-o3 models on
visual reasoning tasks. It implements a light-weight crop-tool loop so the
assistant can request zoomed-in observations using <grounding> tags before
producing a final <answer>.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from verl.trainer.constants import TOOL_CROP_SYSTEM_PROMPT

try:  # Gradio is optional until the GUI is launched
    import gradio as gr
except ImportError:  # pragma: no cover - gradio only required at runtime
    gr = None


LOG_LEVEL = os.getenv("MINIO3_GUI_LOGLEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


@dataclass
class GenerationSettings:
    """Hyper-parameters that control text generation."""

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    max_tool_turns: int = 4

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


_GROUNDING_PATTERN = re.compile(r"<grounding>(.*?)</grounding>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class ParsedResponse:
    raw_text: str
    thought: Optional[str]
    answer: Optional[str]
    groundings: List[Dict]

    @property
    def display_text(self) -> str:
        parts: List[str] = []
        if self.thought:
            parts.append(f"<think>{self.thought.strip()}</think>")
        for grounding in self.groundings:
            parts.append(f"<grounding>{json.dumps(grounding, ensure_ascii=False)}</grounding>")
        if self.answer:
            parts.append(f"<answer>{self.answer.strip()}</answer>")
        if parts:
            return "\n".join(parts)
        return self.raw_text


def _safe_json_loads(payload: str) -> Optional[Dict]:
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return json.loads(payload.replace("'", '"'))
        except json.JSONDecodeError:
            return None


class MiniO3ChatSession:
    """Thin wrapper around Mini-o3 inference with auto crop-tool support."""

    def __init__(
        self,
        model_id: str = "Mini-o3/Mini-o3-7B-v1",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        generation: Optional[GenerationSettings] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.settings = generation or GenerationSettings()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Initializing MiniO3ChatSession model=%s device=%s dtype=%s", model_id, self.device, dtype)

        if dtype.lower() in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype.lower() in {"fp16", "float16"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        logger.info("Loaded processor: %s", self.processor.__class__.__name__)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=trust_remote_code,
        )
        if device != "auto":
            self.model.to(device)
        logger.info("Model weights ready on %s with dtype=%s", self.model.device, self.model.dtype)

        self._messages: List[Dict] = []
        self._image_sources: Dict[str, Image.Image] = {}
        self.reset()

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        logger.debug("Resetting session state")
        self._messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": TOOL_CROP_SYSTEM_PROMPT}],
            }
        ]
        self._image_sources = {}
        self._observation_count = 0

    def add_user_turn(self, text: str, images: Optional[Iterable[Image.Image]] = None) -> None:
        content: List[Dict] = []
        if images:
            image_list = list(images)
            logger.info("Received %d user images", len(image_list))
            for idx, img in enumerate(image_list):
                pil_img = self._ensure_pil_image(img)
                if "original_image" not in self._image_sources:
                    self._image_sources["original_image"] = pil_img
                else:
                    key = f"user_image_{len(self._image_sources)}"
                    self._image_sources[key] = pil_img
                content.append({"type": "image", "image": pil_img})
        if text:
            logger.info("User text: %s", text.strip()[:200])
            content.append({"type": "text", "text": text})
        if not content:
            raise ValueError("User message requires text or at least one image")
        self._messages.append({"role": "user", "content": content})
        logger.debug("Messages in buffer: %d", len(self._messages))

    def run_dialog(self) -> List[ParsedResponse]:
        parsed_responses: List[ParsedResponse] = []
        tool_turn = 0
        logger.info("Starting dialogue loop with max_tool_turns=%d", self.settings.max_tool_turns)
        while tool_turn < self.settings.max_tool_turns:
            parsed = self._generate_once()
            parsed_responses.append(parsed)
            logger.info("Assistant turn=%d thought_present=%s answer_present=%s groundings=%d", tool_turn, bool(parsed.thought), bool(parsed.answer), len(parsed.groundings))
            if parsed.groundings and not parsed.answer:
                tool_turn += 1
                observations = self._materialize_groundings(parsed.groundings)
                if not observations:
                    logger.warning("Grounding requested but no observations were generated")
                    break
                for obs_text, obs_image in observations:
                    self._messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": obs_text},
                                {"type": "image", "image": obs_image},
                            ],
                        }
                    )
                continue
            break
        return parsed_responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_once(self) -> ParsedResponse:
        model_inputs = self._prepare_inputs()
        prompt_length = model_inputs["input_ids"].shape[-1]
        logger.info("Invoking generate with prompt_length=%d max_new_tokens=%d", prompt_length, self.settings.max_new_tokens)
        try:
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=self.settings.do_sample,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                repetition_penalty=self.settings.repetition_penalty,
                use_cache=True,
            )
        except Exception as exc:  # pragma: no cover - surfaced for interactive debugging
            logger.exception("Model generation failed: %s", exc)
            raise
        generated_ids = outputs[:, prompt_length:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info("Assistant raw response: %s", text[:500].strip())
        self._messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
        )
        return self._parse_response(text)

    def _prepare_inputs(self) -> Dict[str, torch.Tensor]:
        prompt = self.processor.apply_chat_template(
            self._messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        images = self._gather_images(self._messages)
        logger.debug("Preparing inputs with %d messages and %d images", len(self._messages), len(images))
        proc_kwargs = {"text": [prompt], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = [images]
        inputs = self.processor(**proc_kwargs)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _gather_images(self, messages: Iterable[Dict]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "image" and item.get("image") is not None:
                    images.append(self._ensure_pil_image(item["image"]))
        logger.debug("Gathered %d images for processor input", len(images))
        return images

    def _parse_response(self, response: str) -> ParsedResponse:
        thought_match = _THINK_PATTERN.search(response)
        answer_match = _ANSWER_PATTERN.search(response)
        groundings: List[Dict] = []
        for match in _GROUNDING_PATTERN.finditer(response):
            payload = _safe_json_loads(match.group(1))
            if isinstance(payload, dict):
                groundings.append(payload)
        thought = thought_match.group(1).strip() if thought_match else None
        answer = answer_match.group(1).strip() if answer_match else None
        parsed = ParsedResponse(
            raw_text=response.strip(),
            thought=thought,
            answer=answer,
            groundings=groundings,
        )
        logger.info("Parsed response summary thought=%s answer=%s groundings=%d", bool(parsed.thought), bool(parsed.answer), len(parsed.groundings))
        return parsed

    def _materialize_groundings(
        self, groundings: Iterable[Dict]
    ) -> List[Tuple[str, Image.Image]]:
        observations: List[Tuple[str, Image.Image]] = []
        for grounding in groundings:
            bbox = grounding.get("bbox_2d")
            source_name = grounding.get("source", "original_image")
            logger.info("Processing grounding from %s with bbox=%s", source_name, bbox)
            if not bbox or len(bbox) != 4:
                logger.warning("Skipping grounding with invalid bbox: %s", bbox)
                continue
            source = self._image_sources.get(source_name)
            if source is None:
                logger.warning("No source image cached under key %s", source_name)
                continue
            crop = self._crop_image(source, bbox)
            if crop is None:
                logger.warning("Failed to crop image for bbox=%s", bbox)
                continue
            self._observation_count += 1
            key = f"observation_{self._observation_count}"
            self._image_sources[key] = crop
            prompt = (
                f"Observation {self._observation_count} generated from {source_name}. "
                "Continue reasoning inside <think>...</think> and put the final answer "
                "inside <answer>...</answer> when you are ready."
            )
            logger.info("Generated observation %s", key)
            observations.append((prompt, crop))
        logger.info("Total observations generated this turn: %d", len(observations))
        return observations

    def _crop_image(
        self, image: Image.Image, bbox: Iterable[float]
    ) -> Optional[Image.Image]:
        pil_image = self._ensure_pil_image(image)
        width, height = pil_image.size
        logger.debug("Cropping image with bbox=%s (source size %sx%s)", bbox, width, height)
        try:
            x0, y0, x1, y1 = [float(x) for x in bbox]
        except (TypeError, ValueError):
            logger.warning("Invalid bbox values: %s", bbox)
            return None
        left = max(0.0, min(1.0, x0)) * width
        top = max(0.0, min(1.0, y0)) * height
        right = max(0.0, min(1.0, x1)) * width
        bottom = max(0.0, min(1.0, y1)) * height
        if right <= left or bottom <= top:
            logger.warning("Degenerate crop computed for bbox=%s -> (%s, %s, %s, %s)", bbox, left, top, right, bottom)
            return None
        logger.debug("Crop box pixels: (%s, %s, %s, %s)", left, top, right, bottom)
        return pil_image.crop((left, top, right, bottom))

    @staticmethod
    def _ensure_pil_image(image: Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB") if image.mode != "RGB" else image
        raise TypeError("Expected PIL.Image input")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _collect_observation_images(sources: Dict[str, Image.Image]) -> List[Image.Image]:
    ordered: List[Tuple[int, Image.Image]] = []
    for key, img in sources.items():
        if not key.startswith("observation_"):
            continue
        try:
            idx = int(key.split("_")[1])
        except (IndexError, ValueError):
            idx = 10**6
        ordered.append((idx, img))
    ordered.sort(key=lambda pair: pair[0])
    images = [img for _, img in ordered]
    logger.debug("Collected %d observation images for UI", len(images))
    return images


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface(session: MiniO3ChatSession) -> "gr.Blocks":
    if gr is None:
        raise ImportError("gradio must be installed to launch the Mini-o3 GUI demo")

    with gr.Blocks() as demo:
        gr.Markdown(
            """# Mini-o3 Visual Reasoning Demo
Upload an image, ask a question, and watch the Mini-o3 model think through
multi-step crops before producing its final answer."""
        )

        state = gr.State({"session": session, "image": None})

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                image_input = gr.Image(type="pil", label="Input image", height=320)
                observation_gallery = gr.Gallery(
                    label="Model-generated observations", height=160, columns=4
                )
                with gr.Accordion("Generation settings", open=False):
                    temperature = gr.Slider(0.0, 1.5, value=session.settings.temperature, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=session.settings.top_p, step=0.05, label="Top-p")
                    max_tokens = gr.Slider(64, 2048, value=session.settings.max_new_tokens, step=32, label="Max new tokens")
                    max_tool_turns = gr.Slider(1, 6, value=session.settings.max_tool_turns, step=1, label="Max tool turns")
                    repetition_penalty = gr.Slider(0.8, 1.5, value=session.settings.repetition_penalty, step=0.01, label="Repetition penalty")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Mini-o3 conversation", height=520)
                user_input = gr.Textbox(placeholder="Ask a question about the image...", label="Your prompt", lines=2)
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    reset_button = gr.Button("Reset conversation")

        def on_image_change(image, state_dict):
            logger.info("Image input updated; resetting session history")
            sess: MiniO3ChatSession = state_dict["session"]
            sess.reset()
            state_dict["image"] = image
            return [], [], state_dict

        image_input.change(
            on_image_change,
            inputs=[image_input, state],
            outputs=[chatbot, observation_gallery, state],
        )

        def respond(
            message: str,
            chat_history: List[Tuple[str, str]],
            state_dict,
            image,
            temperature_value,
            top_p_value,
            max_tokens_value,
            max_tool_turns_value,
            repetition_penalty_value,
        ):
            sess: MiniO3ChatSession = state_dict["session"]
            logger.info("UI trigger: message=%s history_len=%d image_cached=%s", (message or "").strip()[:200], len(chat_history), state_dict.get("image") is not None)
            if not message:
                logger.debug("Empty message received; returning current observations")
                observations = _collect_observation_images(sess._image_sources)
                return chat_history, observations, "", state_dict

            sess.settings.temperature = float(temperature_value)
            sess.settings.top_p = float(top_p_value)
            sess.settings.max_new_tokens = int(max_tokens_value)
            sess.settings.max_tool_turns = int(max_tool_turns_value)
            sess.settings.repetition_penalty = float(repetition_penalty_value)

            if state_dict.get("image") is None:
                if image is None:
                    logger.error("No image supplied; cannot query model")
                    raise gr.Error("Please upload an image before chatting with the model.")
                logger.info("Caching first image and resetting session state")
                sess.reset()
                state_dict["image"] = image
                sess.add_user_turn(message, images=[image])
            else:
                logger.info("Appending text-only user turn")
                sess.add_user_turn(message)

            parsed_turns = sess.run_dialog()
            for idx, parsed in enumerate(parsed_turns):
                user_text = message if idx == 0 else ""
                logger.info("Appending assistant turn %d to chat history", idx)
                chat_history.append((user_text, parsed.display_text))

            observations = _collect_observation_images(sess._image_sources)
            logger.info("Returning %d observations to UI", len(observations))
            return chat_history, observations, "", state_dict

        send_inputs = [
            user_input,
            chatbot,
            state,
            image_input,
            temperature,
            top_p,
            max_tokens,
            max_tool_turns,
            repetition_penalty,
        ]
        send_outputs = [chatbot, observation_gallery, user_input, state]

        send_button.click(respond, inputs=send_inputs, outputs=send_outputs)
        user_input.submit(respond, inputs=send_inputs, outputs=send_outputs)

        def reset_conversation(state_dict):
            logger.info("Reset button pressed; clearing state and image cache")
            sess: MiniO3ChatSession = state_dict["session"]
            sess.reset()
            state_dict["image"] = None
            return None, [], [], state_dict

        reset_button.click(
            reset_conversation,
            inputs=[state],
            outputs=[image_input, chatbot, observation_gallery, state],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-o3 Gradio demo")
    parser.add_argument("--model", default="Mini-o3/Mini-o3-7B-v1", help="Model identifier or local path")
    parser.add_argument("--device", default=None, help="Device to run on (cuda, cpu, or auto)")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype: bfloat16, float16, or float32")
    parser.add_argument("--share", action="store_true", help="Share the public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for the Gradio server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = MiniO3ChatSession(model_id=args.model, device=args.device, dtype=args.dtype)
    demo = build_interface(session)
    demo.queue().launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
