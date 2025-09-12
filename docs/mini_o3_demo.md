# Demo Quickstart

## Input Format
Please refer to our training or validation data to structure your input samples in JSON format. Ensure that the `problem` field includes `"<image>\n"`, and that the `data_source` field is set to one of the following options: `visual_probe_train`, `deepeyes_train`, `visual_probe_easy`, `visual_probe_medium`, `visual_probe_hard`, or `vstar_bench`. These data sources are defined in `verl/utils/reward_score/__init__.py`.
Example format:
```
[
    {
        "images": [
            "[PATH_TO_IMAGE]"
        ],
        "doc_id": "[DOC_ID]",
        "problem": "[PROBLEM]",
        "solution": "[SOLUTION].",
        "data_source": "[DATA_SOURCE]"
    }
]
```

## Inference Commands
You can save the cropped images and reasoning outputs of the model by setting `save_traj=True` and specifying the output directory using `save_traj_dir`.
Please ensure that `DEMO_DATA` refers to your JSON file that follows the required input format.
Below is an example script for performing inference on a single GPU for your reference:
```
export API_KEY=[YOUR_API_KEY]
export API_VERSION=[YOUR_API_VERSION]
export END_POINT=[YOUR_END_POINT]
export BASE_IMAGE_DIR=[YOUR_IMAGES_DIR]
export DEMO_OUTPUT_DIR=[YOUR_OUTPUT_DIR]

VISUALPROBE_TRAIN_DATA=${BASE_IMAGE_DIR}/VisualProbe_train/train.json
DEEPEYES_TRAIN_4K_DATA=${BASE_IMAGE_DIR}/DeepEyes_train_4K/train.json
DEMO_DATA=[YOUR_DEMO_DATA]

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.system_prompt="tool_crop" \
    data.train_files=[${VISUALPROBE_TRAIN_DATA},${DEEPEYES_TRAIN_4K_DATA}] \
    data.val_files=[${DEMO_DATA}] \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.answer_key=solution \
    data.mask_blank=False \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=0 \
    data.tool_call_penalty=0 \
    data.general_qa_reward_fn="general_qa_tool_mc" \
    data.gpt_general_qa_reward_fn="general_qa_tool" \
    data.gpt_extract_answer=True \
    data.extract_answer_tags="strict" \
    data.return_raw_chat=True \
    data.gpt_threads=300 \
    data.tool_call="crop" \
    data.use_tgt_size=False \
    data.max_pixels=2000000 \
    data.min_pixels=40000 \
    reward_model.reward_manager=naive_multithreads_tool \
    actor_rollout_ref.actor.ignore_exceed=True \
    actor_rollout_ref.model.path=Mini-o3/Mini-o3-7B-v1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_generation_round=6 \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={'image': 12}' \
    actor_rollout_ref.rollout.val_max_generation_round=12 \
    'actor_rollout_ref.rollout.val_limit_mm_per_prompt={'image': 12}' \
    actor_rollout_ref.rollout.use_raw_image=True \
    actor_rollout_ref.rollout.multi_turn_prompt_type="v2" \
    actor_rollout_ref.rollout.vllm_infer_batch_size=1 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.save_traj=True \
    actor_rollout_ref.rollout.save_traj_dir=${DEMO_OUTPUT_DIR} \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.rollout.use_relative_coordinates=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='debug' \
    trainer.experiment_name='debug' \
    trainer.val_generations_to_log_to_wandb=512 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=./saves \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.log_training_rollouts_freq=5 \
    trainer.train_generations_to_log_to_wandb=256 \
    trainer.use_3drope=True \
    reward_model.use_hybrid_reward_manager=True \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    actor_rollout_ref.rollout.val_n=1 \
    actor_rollout_ref.rollout.val_do_sample=False \
    trainer.val_only=True
```