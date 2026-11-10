#!/usr/bin/env bash
#SBATCH --job-name=af-full-12h
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
AE="${AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
test -x "$PY"
hostname
date -Is
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
"$PY" -u train_full.py --ae "$AE" --model-cache "$MODEL_CACHE" \
  --output "$PWD/outputs/job-${SLURM_JOB_ID}" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM INT
set +e
wait "$CHILD"
TRAIN_STATUS=$?
set -e
"$PY" export_report.py --output "$PWD/outputs/job-${SLURM_JOB_ID}" --job "$SLURM_JOB_ID"
printf 'RESULT_DIRECTORY=%s/outputs/job-%s\n' "$PWD" "$SLURM_JOB_ID"
date -Is
exit "$TRAIN_STATUS"
