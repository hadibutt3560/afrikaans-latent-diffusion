#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"

: "${LATENT_MODEL_PATH:?Set LATENT_MODEL_PATH to an autoencoder output containing args.json and model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rocstories/diffusion-smoke}"
STEPS="${STEPS:-50}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

python train_text_diffusion.py \
  --dataset_name roc \
  --enc_dec_model facebook/bart-base \
  --latent_model_path "${LATENT_MODEL_PATH}" \
  --require_latent_model \
  --max_seq_len 64 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --learning_rate 2e-4 \
  --num_train_steps "${STEPS}" \
  --lr_schedule cosine \
  --lr_warmup_steps 10 \
  --tx_dim 64 \
  --tx_depth 1 \
  --objective pred_v \
  --loss_type l2 \
  --train_schedule cosine \
  --sampling_timesteps 10 \
  --sampler ddpm \
  --self_condition \
  --train_prob_self_cond 0.5 \
  --scale_shift \
  --num_dense_connections 0 \
  --ema_decay 0.9999 \
  --clip_grad_norm 1.0 \
  --num_samples 4 \
  --save_and_sample_every "${STEPS}" \
  --skip_expensive_metrics \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_name roc-diffusion-smoke

if [[ "${RUN_RESUME_TEST:-0}" == "1" ]]; then
  python train_text_diffusion.py \
    --resume_training \
    --resume_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --resume_to_step "$((STEPS + 10))" \
    --skip_expensive_metrics
fi
