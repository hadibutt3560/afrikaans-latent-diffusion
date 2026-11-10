#!/usr/bin/env bash
#SBATCH --job-name=af-decoder-refine
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster106
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
AE="${AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
OUT="$PWD/outputs/job-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
test -f "$SIF"
test -x "$PY"
test -f "$AE/model.pt"
test -d "$MODEL_CACHE"
hostname
date -Is
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
# Do not reinterpret local CUDA_VISIBLE_DEVICES as proof of GPU health.
# The Python preflight performs CUDA, weight, cache and baseline checks.
apptainer exec --nv --bind "$HOME:$HOME,$PWD:$PWD,$BASE:$BASE" --pwd "$PWD" \
  "$SIF" "$PY" -u train_decoder.py --ae "$AE" --model-cache "$MODEL_CACHE" \
  --output "$OUT" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM INT
wait "$CHILD"
printf 'RESULT_DIRECTORY=%s\n' "$OUT"
date -Is
