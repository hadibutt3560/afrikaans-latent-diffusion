#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/rocstories/autoencoder-smoke}"
STEPS="${STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

python train_latent_model.py \
  --dataset_name roc \
  --enc_dec_model facebook/bart-base \
  --max_seq_len 64 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_encoder_latents 32 \
  --num_decoder_latents 32 \
  --dim_ae 64 \
  --num_layers 3 \
  --l2_normalize_latents \
  --learning_rate 1e-4 \
  --lr_schedule linear \
  --lr_warmup_steps 10 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-2 \
  --num_train_steps "${STEPS}" \
  --eval_every "${STEPS}" \
  --validation_max_examples 32 \
  --skip_expensive_metrics \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_name roc-ae-smoke

if [[ "${RUN_RESUME_TEST:-0}" == "1" ]]; then
  python train_latent_model.py \
    --resume_training \
    --resume_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --resume_to_step "$((STEPS + 10))" \
    --skip_expensive_metrics \
    --validation_max_examples 32
fi
