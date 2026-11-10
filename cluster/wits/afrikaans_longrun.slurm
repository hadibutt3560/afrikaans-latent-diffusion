#!/usr/bin/env bash
#SBATCH --job-name=af-longrun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@600
# Account, partition, resource request, walltime and output are supplied by submit_longrun.sh.
set -euo pipefail
: "${REPO:?}" "${WORKSPACE:?}" "${ACTION:?}" "${RUN_SECONDS:?}"
: "${SIF:?}" "${PY:?}" "${AE:?}" "${DATA:?}" "${MODEL_CACHE:?}" "${CONFIG:?}"
cd "$REPO"
test -f "$SIF"
test -x "$PY"
mkdir -p "$WORKSPACE"
# Kernel-managed advisory lock is released even if this job is killed.
exec 9>"$WORKSPACE/.job.lock"
flock -n 9 || { echo "Another job is writing this workspace" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
hostname
date -u
nvidia-smi
# Shared paths may be outside HOME; bind every explicit input/output root.
BIND="$HOME:$HOME,$REPO:$REPO,$WORKSPACE:$WORKSPACE,$DATA:$DATA,$AE:$AE,$MODEL_CACHE:$MODEL_CACHE,$HF_HOME:$HF_HOME"
apptainer exec --nv --bind "$BIND" --pwd "$REPO" "$SIF" "$PY" -u tools/afrikaans_longrun.py "$ACTION" \
  --config "$CONFIG" --workspace "$WORKSPACE" --data "$DATA" --ae "$AE" \
  --model-cache "$MODEL_CACHE" --seconds "$RUN_SECONDS" &
CHILD=$!
INTERRUPTED=0
forward_stop() { INTERRUPTED=1; kill -TERM "$CHILD" 2>/dev/null || true; }
trap forward_stop USR1 TERM INT
set +e
# A trapped signal interrupts Bash wait before the child has saved. Keep waiting.
while true; do
  wait "$CHILD"
  RC=$?
  if [[ "$INTERRUPTED" = 1 ]]; then
    INTERRUPTED=0
    continue
  fi
  break
done
set -e
if [[ "$RC" = 75 ]]; then
  echo "Clean checkpoint boundary reached. Resubmit the same action/workspace."
  exit 0  # afterok chains may resume; genuine failures retain non-zero status.
fi
exit "$RC"
