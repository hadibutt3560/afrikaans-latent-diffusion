#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export HF_HOME=/home/scientist/.cache/huggingface
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OUT=outputs/nchlt-afrikaans/recovery/pilot-decoder_last-no-backbone-dropout-500
BASE=outputs/nchlt-afrikaans/recovery/baseline-step6000
PY=/home/scientist/miniconda3/envs/ld4lg-reproduction/bin/python

if [ -e "$OUT" ]; then
    echo "Output already exists. Stop here to avoid overwriting progress."
    exit 1
fi
test -x "$PY"
test -f "$BASE/model.pt"
mkdir -p "$OUT"

"$PY" -u train_latent_model.py \
  --dataset_name nchlt_afrikaans \
  --enc_dec_model google/mt5-base \
  --init_dir "$BASE" \
  --output_dir "$OUT" \
  --lm_mode decoder_last \
  --backbone_eval_mode \
  --max_seq_len 128 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_encoder_latents 32 \
  --num_decoder_latents 32 \
  --dim_ae 64 \
  --num_layers 3 \
  --l2_normalize_latents \
  --learning_rate 0.0001 \
  --lr_schedule constant_with_warmup \
  --lr_warmup_steps 25 \
  --num_train_steps 500 \
  --eval_every 250 \
  --validation_max_examples 128 \
  --validation_shuffle \
  --skip_expensive_metrics \
  --fail_on_validation_error \
  --dataloader_num_workers 0 \
  --mixed_precision bf16 \
  --seed 42 2>&1 | tee "$OUT/console.log"
