#!/usr/bin/env bash
# Local launcher integration check. All cluster/container commands are fake.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT=$(mktemp -d)
# No recursive cleanup: retain tiny test fixtures for debugging.
export REPO="$ROOT" WORKSPACE="$TEST_ROOT/workspace" ACTION=run RUN_SECONDS=30
export SIF="$TEST_ROOT/image.sif" PY="$(command -v bash)" AE="$TEST_ROOT" DATA="$TEST_ROOT"
export MODEL_CACHE="$TEST_ROOT" CONFIG="$ROOT/configs/afrikaans_longrun.json" HF_HOME="$TEST_ROOT"
mkdir -p "$WORKSPACE"
touch "$SIF"
export READY_FILE="$TEST_ROOT/ready" SIGNAL_FILE="$TEST_ROOT/saved"
nvidia-smi() { return 0; }
flock() { return 0; }
apptainer() {
  trap 'printf "checkpoint saved\n" > "$SIGNAL_FILE"; exit 75' TERM
  touch "$READY_FILE"
  while true; do sleep 0.1; done
}
export -f nvidia-smi flock apptainer
bash "$ROOT/cluster/wits/afrikaans_longrun.slurm" > "$TEST_ROOT/launcher.log" 2>&1 &
LAUNCHER=$!
for attempt in {1..100}; do
  [[ -f "$READY_FILE" ]] && break
  sleep 0.05
done
[[ -f "$READY_FILE" ]] || { echo "Fake child did not start" >&2; exit 1; }
kill -USR1 "$LAUNCHER"
wait "$LAUNCHER"
test -f "$SIGNAL_FILE"
printf 'PASS: advance signal forwarded, checkpoint child awaited, clean pause returned success\n'
# Genuine child errors must fail the launcher, rather than continuing a chain.
apptainer() { return 23; }
export -f apptainer
set +e
bash "$ROOT/cluster/wits/afrikaans_longrun.slurm" > "$TEST_ROOT/failure.log" 2>&1
RC=$?
set -e
[[ "$RC" = 23 ]]
printf 'PASS: genuine child failure retained exit code 23\n'
