#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH:?Wits must provide SCRATCH, or set it to your allocated scratch directory}"
export LD4LG_SCRATCH_ROOT="${LD4LG_SCRATCH_ROOT:-${SCRATCH}/ld4lg-${USER}}"
export HF_HOME="${HF_HOME:-${LD4LG_SCRATCH_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${LD4LG_SCRATCH_ROOT}/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p \
  "${LD4LG_SCRATCH_ROOT}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${WANDB_DIR}" \
  "${LD4LG_SCRATCH_ROOT}/outputs" \
  "${LD4LG_SCRATCH_ROOT}/logs"
