#!/usr/bin/env bash
#SBATCH --job-name=swahili72
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --nodelist=mscluster110
#SBATCH --signal=B:USR1@900
#SBATCH --output=/datasets/hbutt/swahili72/slurm/%x-%j.log
set -euo pipefail
ROOT=/datasets/hbutt/swahili72/package
OUT=/datasets/hbutt/swahili72/results
SIF="$HOME/containers/anaconda-2025-12-2.sif"
PY=/datasets/hbutt/swahili72/env/bin/python
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/datasets/hbutt/swahili72/hf-cache
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
mkdir -p "$HF_HOME"
exec 9>/datasets/hbutt/swahili72/.workflow.lock
flock -n 9 || { echo 'Another Swahili workflow is active'; exit 1; }
# Scheduler end time is authoritative; allocating less time never expands the internal budget.
END="$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')"
export SW_ALLOCATION_END="$(date -d "$END" +%s)"
hostname; date -u; nvidia-smi
ARGS=()
if [[ "${1:-}" == --resume ]]; then ARGS+=(--resume); elif [[ $# -ne 0 ]]; then exit 2; fi
apptainer exec --nv --bind "$HOME:$HOME,/datasets:/datasets" --pwd "$ROOT" "$SIF" "$PY" -u "$ROOT/run.py" --root "$ROOT" --out "$OUT" "${ARGS[@]}" &
CHILD=$!
forward_stop() { kill -TERM "$CHILD" 2>/dev/null || true; }
trap forward_stop USR1 TERM INT
set +e
while true; do
  wait "$CHILD"; RC=$?
  kill -0 "$CHILD" 2>/dev/null || break
done
exit "$RC"
