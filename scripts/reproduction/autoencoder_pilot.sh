#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/rocstories/autoencoder-pilot}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

python train_latent_model.py \
  --dataset_name roc \
  --enc_dec_model facebook/bart-base \
  --max_seq_len 64 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --num_encoder_latents 32 \
  --num_decoder_latents 32 \
  --dim_ae 64 \
  --num_layers 3 \
  --l2_normalize_latents \
  --learning_rate 1e-4 \
  --lr_schedule linear \
  --lr_warmup_steps 500 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-2 \
  --num_train_steps 5000 \
  --eval_every 1000 \
  --validation_max_examples 1000 \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_name roc-ae-pilot
