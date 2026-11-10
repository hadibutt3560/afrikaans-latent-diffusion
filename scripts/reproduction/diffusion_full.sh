#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"

: "${LATENT_MODEL_PATH:?Set LATENT_MODEL_PATH to the validated full autoencoder output}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rocstories/diffusion-full}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# One-GPU default: 8 examples/GPU x 16 accumulation x 1 GPU = global batch 128.
python train_text_diffusion.py \
  --dataset_name roc \
  --enc_dec_model facebook/bart-base \
  --latent_model_path "${LATENT_MODEL_PATH}" \
  --require_latent_model \
  --max_seq_len 64 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-16}" \
  --learning_rate 2e-4 \
  --num_train_steps 250000 \
  --lr_schedule cosine \
  --lr_warmup_steps 1000 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-6 \
  --tx_dim 768 \
  --tx_depth 12 \
  --objective pred_v \
  --loss_type l2 \
  --train_schedule cosine \
  --sampling_timesteps 250 \
  --sampler ddpm \
  --self_condition \
  --train_prob_self_cond 0.5 \
  --scale_shift \
  --num_dense_connections 3 \
  --ema_decay 0.9999 \
  --ema_update_every 1 \
  --clip_grad_norm 1.0 \
  --num_samples 1000 \
  --save_and_sample_every 5000 \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_name roc-diffusion-full
