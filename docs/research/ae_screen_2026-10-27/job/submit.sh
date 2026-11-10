#!/usr/bin/env bash
# Submit the three chained jobs once. Run from the extracted bundle directory on the login node.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
OLD_AE="${OLD_AE:-$BASE/outputs/afrikaans-24h-56531/best}"
MODEL_CACHE="${MODEL_CACHE:-$BASE/models/mt5-base}"
MGPT="${MGPT:-$HOME/afrikaans-decoder-dropout-final-20261025/evaluation/mgpt}"
test -x "$PY" || { echo "Missing Python environment: $PY" >&2; exit 1; }
test -f "$OLD_AE/model.pt" || { echo "Missing old autoencoder: $OLD_AE" >&2; exit 1; }
test -d "$MODEL_CACHE" || { echo "Missing mT5 cache: $MODEL_CACHE" >&2; exit 1; }
test -f "$MGPT/pytorch_model.bin" || { echo "Missing pinned mGPT: $MGPT" >&2; exit 1; }
test ! -e submitted-job-ids.txt || { echo 'Already submitted; see submitted-job-ids.txt' >&2; exit 1; }
mkdir -p logs
echo 'Checking package hashes...'
sha256sum -c bundle-manifest.sha256 > logs/package-check.txt
export PY BASE OLD_AE MODEL_CACHE MGPT
J1=$(sbatch --parsable job/screen.slurm)
J2=$(sbatch --parsable --dependency=afterok:"${J1%%;*}" job/diffusion_train.slurm)
J3=$(sbatch --parsable --dependency=afterany:"${J2%%;*}" job/diffusion_finish.slurm)
printf 'screen=%s\ndiffusion_train=%s\ndiffusion_finish=%s\n' "${J1%%;*}" "${J2%%;*}" "${J3%%;*}" | tee submitted-job-ids.txt
squeue -u "$USER" -o '%.18i %.22j %.10T %.10M %.30R'
