#!/usr/bin/env bash
#SBATCH --job-name=af-joint-final
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster110
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
AE="${AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
OUT="$PWD/outputs/job-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
CHILD=''
finish() {
  status=$?
  trap - EXIT
  set +e
  "$PY" export_report.py --output "$OUT" --job "$SLURM_JOB_ID" --exit-code "$status"
  report_status=$?
  printf 'RESULT_DIRECTORY=%s\nTRAIN_EXIT=%s\nEXPORT_EXIT=%s\n' "$OUT" "$status" "$report_status"
  date -Is
  if [ "$status" -ne 0 ]; then exit "$status"; fi
  exit "$report_status"
}
stop() {
  if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; fi
  exit 143
}
trap finish EXIT
trap stop TERM INT
test -x "$PY"
hostname
date -Is
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
mkdir -p "$(dirname "$OUT")"
scontrol show job "$SLURM_JOB_ID" > "outputs/scheduler-${SLURM_JOB_ID}.txt" || true
"$PY" -u train_joint.py --ae "$AE" --model-cache "$MODEL_CACHE" --output "$OUT" &
CHILD=$!
set +e
wait "$CHILD"
status=$?
CHILD=''
exit "$status"
