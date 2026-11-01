#!/usr/bin/env bash
# Mock orchestration check: no cluster access or real Python/model execution.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE=$(mktemp -d)
mkdir -p "$FIXTURE/cluster/wits" "$FIXTURE/work/cache"
printf '{"complete": true}\n' > "$FIXTURE/work/cache/manifest.json"
{
  printf 'export REPO=%q\n' "$FIXTURE"
  printf 'export WORKSPACE=%q\n' "$FIXTURE/work"
  printf 'export DATA=%q AE=%q MODEL_CACHE=%q HF_HOME=%q\n' "$FIXTURE" "$FIXTURE" "$FIXTURE" "$FIXTURE"
  printf 'export SIF=mock.sif PY=mock-python CONFIG=mock-config\n'
} > "$FIXTURE/cluster/wits/longrun.env"
export SLURM_SUBMIT_DIR="$FIXTURE" CALL_LOG="$FIXTURE/calls"
nvidia-smi() { return 0; }
flock() { return 0; }
apptainer() {
  local mode=
  for arg in "$@"; do
    case "$arg" in smoke|benchmark|plan) mode="$arg";; esac
  done
  printf '%s\n' "$mode" >> "$CALL_LOG"
  if [[ "${FAIL_MODE:-}" = "$mode" ]]; then return 23; fi
  printf '{"mock": true}\n' > "$WORKSPACE/$mode.json"
}
export -f nvidia-smi flock apptainer
bash "$ROOT/cluster/wits/longrun_readiness.slurm" > "$FIXTURE/success.log" 2>&1
[[ "$(cat "$CALL_LOG")" = $'smoke\nbenchmark\nplan' ]]
test -s "$FIXTURE/work/plan.json"
printf 'PASS: smoke, benchmark, plan execute in order; no training stage\n'
# A new workspace isolates the failing attempt without deleting evidence.
mkdir -p "$FIXTURE/failure/cache"
printf '{"complete": true}\n' > "$FIXTURE/failure/cache/manifest.json"
printf 'export WORKSPACE=%q\n' "$FIXTURE/failure" >> "$FIXTURE/cluster/wits/longrun.env"
export FAIL_MODE=smoke CALL_LOG="$FIXTURE/failure-calls"
set +e
bash "$ROOT/cluster/wits/longrun_readiness.slurm" > "$FIXTURE/failure.log" 2>&1
RC=$?
set -e
[[ "$RC" = 23 && "$(cat "$CALL_LOG")" = smoke ]]
test ! -e "$FIXTURE/failure/benchmark.json"
printf 'PASS: smoke failure stops benchmark and planning\n'
