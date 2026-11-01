#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
AE="${AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
test -f "$SIF" || { printf 'Missing container: %s\n' "$SIF" >&2; exit 1; }
test -x "$PY" || { printf 'Missing Python environment: %s\n' "$PY" >&2; exit 1; }
test -f "$AE/model.pt" || { printf 'Missing selected AE: %s\n' "$AE/model.pt" >&2; exit 1; }
test -d "$MODEL_CACHE" || { printf 'Missing model cache: %s\n' "$MODEL_CACHE" >&2; exit 1; }
if [[ -e submitted-job-id.txt ]]; then
  echo 'This package has already submitted a job; inspect submitted-job-id.txt and its logs. Do not submit duplicates.' >&2
  exit 1
fi
mkdir -p logs
sha256sum -c bundle-manifest.sha256 > logs/bundle-verification.txt
timeout 7200 apptainer exec --bind "$HOME:$HOME,$PWD:$PWD,$BASE:$BASE" --pwd "$PWD" \
  "$SIF" "$PY" -u train_decoder.py --cpu-only --ae "$AE" --model-cache "$MODEL_CACHE" \
  > logs/cpu-preflight.txt 2>&1 || { cat logs/cpu-preflight.txt; exit 1; }
# Prevent queueing an experiment after the agreed start deadline.
apptainer exec "$SIF" "$PY" -c 'from datetime import datetime; import time; assert time.time() < datetime.fromisoformat("2026-10-24T15:00:00+02:00").timestamp(), "Start deadline has passed; do not submit"'
export SIF PY BASE AE MODEL_CACHE
JOB_ID=$(sbatch --parsable decoder_refine.slurm)
printf '%s\n' "$JOB_ID" > submitted-job-id.txt
printf 'Submitted job %s\n' "$JOB_ID"
squeue -j "${JOB_ID%%;*}" -o '%.18i %.8T %.10M %.30R'
