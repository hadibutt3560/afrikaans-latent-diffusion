#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_DIR="${DATASET_DIR:-data/nchlt_afrikaans/processed/stage_5_7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/nchlt-afrikaans/autoencoder-smoke}"
MODEL_NAME="${MODEL_NAME:-google/mt5-base}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
STEPS="${STEPS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

python train_latent_model.py \
  --dataset_name nchlt_afrikaans \
  --dataset_dir "${DATASET_DIR}" \
  --enc_dec_model "${MODEL_NAME}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
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
  --lr_warmup_steps 1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-2 \
  --num_train_steps "${STEPS}" \
  --eval_every "${STEPS}" \
  --validation_max_examples "${EVAL_EXAMPLES}" \
  --skip_expensive_metrics \
  --dataloader_num_workers 0 \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_name nchlt-afr-mt5-ae-smoke

if [[ "${RUN_RESUME_TEST:-0}" == "1" ]]; then
  python train_latent_model.py \
    --resume_training \
    --resume_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --resume_to_step "$((STEPS + 1))" \
    --skip_expensive_metrics \
    --validation_max_examples "${EVAL_EXAMPLES}"
fi
