#!/usr/bin/env bash
# Shared environment for the three jobs. Sourced, not executed.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
OLD_AE="${OLD_AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
MGPT="${MGPT:-$HOME/afrikaans-decoder-dropout-final-20261025/evaluation/mgpt}"
OUT="$PWD/outputs/screen"
export BUNDLE_ROOT="$PWD" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD:$PWD/vendor:$PWD/job"
mkdir -p "$OUT" logs
hostname
date -Is
# Fail in seconds, not hours, on a node whose GPU driver is broken (seen on mscluster110/112).
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
"$PY" -c 'import torch; assert torch.cuda.is_available(); x = torch.randn(512, 512, device="cuda"); assert torch.isfinite(x @ x.T).all(); print(torch.cuda.get_device_name())'
scontrol show job "$SLURM_JOB_ID" > "logs/scheduler-${SLURM_JOB_ID}.txt" || true

CHILD=''
# TERM goes to the whole process group: the orchestrator ignores it and waits, while the training
# subprocess (train_ae.py or tools/afrikaans_longrun.py) saves a checkpoint and exits 75.
forward_stop() { if [ -n "$CHILD" ]; then kill -TERM -- "-$CHILD" 2>/dev/null || true; fi; }
trap forward_stop TERM INT

run_child() {
  setsid "$@" &
  CHILD=$!
  set +e
  while true; do
    wait "$CHILD"; status=$?
    if kill -0 "$CHILD" 2>/dev/null; then continue; fi
    break
  done
  set -e
  CHILD=''
  return "$status"
}

export_results() {
  local name="$1"
  local archive="$PWD/results-${name}-${SLURM_JOB_ID}.tar.gz"
  # Checkpoints and latent caches stay on the cluster; everything else is exported.
  tar --exclude='*.pt' --exclude='*.pt.tmp' --exclude='outputs/screen/diffusion/cache' -czf "$archive" outputs logs || true
  sha256sum "$archive" > "$archive.sha256" || true
  printf 'RESULT_ARCHIVE=%s\n' "$archive"
}
