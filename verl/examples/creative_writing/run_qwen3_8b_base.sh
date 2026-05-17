#!/bin/bash
# IP-GRM GRPO training — Qwen3-8B-Base with two-stage reward model
set -e

# ── Paths (override via environment) ──────────────────────────────────────
ACTOR_MODEL=${ACTOR_MODEL:-"Qwen/Qwen3-8B-Base"}
REWARD_MODEL=${REWARD_MODEL:-"BBQGOD/DeepSeek-GRM-16B"}
TRAIN_DATA=${TRAIN_DATA:-""}          # parquet path
PROCESS_FN=${PROCESS_FN:-"$(dirname "$0")/ipgrm_process_fn.py"}

# ── Wandb ─────────────────────────────────────────────────────────────────
export WANDB_API_KEY=${WANDB_API_KEY:?"Set WANDB_API_KEY"}
export WANDB_RESUME=allow
export WANDB_RUN_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")

# ── Ray cluster ───────────────────────────────────────────────────────────
export NCCL_TIMEOUT=1800
ray stop --force
ray start --head --disable-usage-stats --include-dashboard=False --num-gpus=8 --num-cpus=64

echo "Actor model : $ACTOR_MODEL"
echo "Reward model: $REWARD_MODEL"
echo "Run ID      : $WANDB_RUN_ID"

# ── Launch training ───────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=15000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.27 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model=vllm_reward_model \
    +reward_model.style='model' \
    +reward_model.enable_self_eval_penalty=False \
    +reward_model.input_model_config.path="$ACTOR_MODEL" \
    +reward_model.data_processor_config.strip_think_tag=True \
    reward_model.model_config.path="$REWARD_MODEL" \
    reward_model.tensor_model_parallel_size=1 \
    reward_model.micro_batch_size_per_gpu=4 \
    reward_model.max_num_seqs=32 \
    reward_model.max_num_batched_tokens=800000 \
    reward_model.gpu_memory_utilization=0.55 \
    reward_model.prompt_length=20000 \
    reward_model.response_length=15000 \
    reward_model.max_model_len=45000 \
    reward_model.enable_chunked_prefill=True \
    reward_model.enable_prefix_caching=True \
    reward_model.enforce_eager=True \
    reward_model.data_processor_config.path="$PROCESS_FN" \
    reward_model.data_processor_config.preprocess_fn_name=construct_cached_principles_input \
    reward_model.data_processor_config.postprocess_fn_name=parse_reward_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='ipgrm_creative_writing' \
    trainer.experiment_name="qwen3_8b_base_ipgrm" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.resume_mode=auto "$@"

ray stop --force
