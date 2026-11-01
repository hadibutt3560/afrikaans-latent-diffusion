#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
export MATCH="${MATCH:-outputs/nchlt-afrikaans/diffusion-50k-retry-20260920}"
PY="$HOME/.conda/envs/ld4lg-blackwell/bin/python"
test -f "$SIF"
test -x "$PY"
# CPU-only preflight: check local tokenizer, source protocol and data before queuing.
apptainer exec --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" \
 tools/train_afrikaans_baseline.py --match-run "$MATCH" --output-dir outputs/unused-preflight --check-only
mkdir -p "$HOME/cluster-logs"
job=$(sbatch --parsable --output="$HOME/cluster-logs/af-baseline-%j.log" cluster/wits/afrikaans_baseline.slurm)
printf '%s\n' "$job" | tee "baseline-submission-${job}.txt"
echo "Submitted baseline training: $job. It will run after logout."
squeue -j "$job"
