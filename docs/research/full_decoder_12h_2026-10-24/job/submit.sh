#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
AE="${AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
test -x "$PY" || { echo 'Missing Python environment' >&2; exit 1; }
test -f "$AE/model.pt" || { echo 'Missing original AE checkpoint' >&2; exit 1; }
test -d "$MODEL_CACHE" || { echo 'Missing original model cache' >&2; exit 1; }
test ! -e submitted-job-id.txt || { echo 'Already submitted; inspect the saved job ID.' >&2; exit 1; }
mkdir -p logs
echo 'Checking package hashes; this can take a minute...'
sha256sum -c bundle-manifest.sha256 > logs/package-check.txt
"$PY" -c 'from datetime import datetime; import time; assert time.time() < datetime.fromisoformat("2026-10-26T00:00:00+02:00").timestamp(), "Latest start window has passed"'
export PY BASE AE MODEL_CACHE
# No training/import-heavy preflight on the login node. Checks run in the allocation.
JOB_ID=$(sbatch --parsable full_decoder.slurm)
printf '%s\n' "$JOB_ID" > submitted-job-id.txt
printf 'Submitted job %s\n' "$JOB_ID"
squeue -j "${JOB_ID%%;*}" -o '%.18i %.20j %.10T %.10M %.30R'
