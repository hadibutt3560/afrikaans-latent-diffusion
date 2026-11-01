#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

LATENT_MODEL_PATH="${LATENT_MODEL_PATH:-outputs/cluster-results/afrikaans-56531/best}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/nchlt-afrikaans/diffusion-smoke}"
STEPS="${STEPS:-50}"

test -f "$LATENT_MODEL_PATH/model.pt"
test -f "$LATENT_MODEL_PATH/args.json"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite $OUTPUT_DIR" >&2
  exit 1
fi

python -u train_text_diffusion.py \
  --dataset_name nchlt_afrikaans \
  --dataset_dir data/nchlt_afrikaans/processed/stage_5_7 \
  --enc_dec_model google/mt5-base \
  --latent_model_path "$LATENT_MODEL_PATH" \
  --require_latent_model \
  --max_seq_len 128 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
  --learning_rate 2e-4 \
  --num_train_steps "$STEPS" \
  --lr_schedule cosine \
  --lr_warmup_steps 5 \
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
  --ema_decay 0.9999 \
  --clip_grad_norm 1.0 \
  --num_samples 4 \
  --save_and_sample_every "$STEPS" \
  --skip_expensive_metrics \
  --mixed_precision "${MIXED_PRECISION:-no}" \
  --dataloader_num_workers 0 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR"
