#!/usr/bin/env bash
#SBATCH --job-name=af-ready
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
# Submit with the account, partition, CPUs, memory and RESOURCE_ARGS in longrun.env.
# Runs existing, hashed Python unchanged so the prepared cache stays compatible.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the longrun repository}"
source cluster/wits/longrun.env
mkdir -p "$WORKSPACE/slurm" "$HF_HOME"
exec 9>"$WORKSPACE/.job.lock"
flock -n 9 || { echo "Another job is writing this workspace" >&2; exit 1; }
test -f "$WORKSPACE/cache/manifest.json"
grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$WORKSPACE/cache/manifest.json" || {
  echo "Preparation is incomplete; finish prepare first" >&2; exit 1;
}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BIND="$HOME:$HOME,$REPO:$REPO,$WORKSPACE:$WORKSPACE,$DATA:$DATA,$AE:$AE,$MODEL_CACHE:$MODEL_CACHE,$HF_HOME:$HF_HOME"
DEADLINE=$(( $(date +%s) + ${READINESS_SECONDS:-6300} ))
STOP_REQUESTED=0
WAIT_INTERRUPTED=0
CHILD=
forward_stop() {
  STOP_REQUESTED=1
  WAIT_INTERRUPTED=1
  if [[ -n "$CHILD" ]]; then kill -TERM "$CHILD" 2>/dev/null || true; fi
}
trap forward_stop USR1 TERM INT
hostname
nvidia-smi
for ACTION in smoke benchmark plan; do
  if (( STOP_REQUESTED )); then exit 75; fi
  REMAINING=$(( DEADLINE - $(date +%s) ))
  (( REMAINING > 0 )) || { echo "Readiness job budget exhausted" >&2; exit 75; }
  # The Python entry points validate inputs and reject stale/incompatible protocols.
  # Do not silently reuse readiness evidence from another GPU/configuration.
  if [[ -e "$WORKSPACE/$ACTION.json" ]]; then
    echo "$ACTION.json already exists. Inspect previous results before rerunning readiness." >&2
    exit 1
  fi
  printf '\nStarting %s; %s seconds remain in this allocation\n' "$ACTION" "$REMAINING"
  apptainer exec --nv --bind "$BIND" --pwd "$REPO" "$SIF" "$PY" -u tools/afrikaans_longrun.py "$ACTION" \
    --config "$CONFIG" --workspace "$WORKSPACE" --data "$DATA" --ae "$AE" \
    --model-cache "$MODEL_CACHE" --seconds "$REMAINING" &
  CHILD=$!
  set +e
  while true; do
    wait "$CHILD"
    RC=$?
    if (( WAIT_INTERRUPTED )); then WAIT_INTERRUPTED=0; continue; fi
    break
  done
  set -e
  CHILD=
  (( RC == 0 )) || exit "$RC"
  (( STOP_REQUESTED == 0 )) || exit 75
  test -s "$WORKSPACE/$ACTION.json"
done
echo "READINESS COMPLETE: inspect smoke.json, benchmark.json and plan.json."
echo "No long training job has been submitted."
