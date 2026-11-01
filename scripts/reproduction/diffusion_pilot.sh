#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"

: "${LATENT_MODEL_PATH:?Set LATENT_MODEL_PATH to the validated autoencoder output}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rocstories/diffusion-pilot}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

python train_text_diffusion.py \
  --dataset_name roc \
  --enc_dec_model facebook/bart-base \
  --latent_model_path "${LATENT_MODEL_PATH}" \
  --require_latent_model \
  --max_seq_len 64 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate 2e-4 \
  --num_train_steps "${STEPS:-5000}" \
  --lr_schedule cosine \
  --lr_warmup_steps 500 \
  --tx_dim 256 \
  --tx_depth 4 \
  --objective pred_v \
  --loss_type l2 \
  --train_schedule cosine \
  --sampling_timesteps 50 \
  --sampler ddpm \
  --self_condition \
  --train_prob_self_cond 0.5 \
  --scale_shift \
  --num_dense_connections 1 \
  --ema_decay 0.9999 \
  --clip_grad_norm 1.0 \
  --num_samples 100 \
  --save_and_sample_every 1000 \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_name roc-diffusion-pilot
