#!/bin/bash
# SFT fine-tuning for DeepSeek-GRM reward model (FSDP)
set -x

# ── Paths (override via environment) ──────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"BBQGOD/DeepSeek-GRM-16B"}
TRAIN_DATA=${TRAIN_DATA:-"path/to/sft_train.parquet"}
VAL_DATA=${VAL_DATA:-"path/to/sft_val.parquet"}
SAVE_PATH=${SAVE_PATH:-"./checkpoints/deepseek-grm-sft"}

# ── Wandb ─────────────────────────────────────────────────────────────────
export WANDB_API_KEY=${WANDB_API_KEY:?"Set WANDB_API_KEY"}

# ── Training hyperparameters ──────────────────────────────────────────────
NPROC_PER_NODE=8
MAX_LENGTH=63840
MICRO_BS=1
LR=1e-5
EPOCHS=5
SAVE_FREQ=18
TEST_FREQ=1
SEED=42

# ── Launch ────────────────────────────────────────────────────────────────
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=$MAX_LENGTH \
    data.micro_batch_size_per_gpu=$MICRO_BS \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.use_liger=false \
    optim.lr=$LR \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name='ipgrm_sft' \
    trainer.experiment_name='deepseek_grm_sft' \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.seed=$SEED \
    trainer.max_ckpt_to_keep=10 \
    trainer.logger='["console","wandb"]' "$@"
