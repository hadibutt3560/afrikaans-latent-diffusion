#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SIF=/usr/local/containers/anaconda/anaconda-2025-12-2.sif
PY="$HOME/.conda/envs/ld4lg-blackwell/bin/python"
test -f "$SIF"
test -x "$PY"
command -v apptainer
sha256sum --check bundle-manifest.sha256
# Check CPU imports before joining the queue; GPU checks run inside the allocation.
apptainer exec --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -u tools/wits_afrikaans_train.py --check-only
mkdir -p "$HOME/cluster-logs"
job=$(sbatch --parsable --output="$HOME/cluster-logs/afrikaans-%j.log" cluster/wits/afrikaans_24h.slurm)
printf '%s\n' "$job" | tee submitted-job-id.txt
echo "Training job submitted. This job checks the GPU and trains in the same allocation."
squeue -j "$job"
