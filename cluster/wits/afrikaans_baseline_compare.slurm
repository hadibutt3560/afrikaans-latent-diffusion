#!/usr/bin/env bash
#SBATCH --job-name=af-ar-compare
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster107,mscluster108,mscluster111
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
PY="$HOME/.conda/envs/ld4lg-blackwell/bin/python"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
test -n "${BASELINE_DIR:?Set BASELINE_DIR to the completed baseline directory}"
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -c \
 'import torch; assert torch.cuda.is_available(); print(torch.ones(8,device="cuda").sum().item())'
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -u \
 tools/compare_afrikaans_generators.py --baseline-dir "$BASELINE_DIR" \
 --diffusion-samples "${DIFFUSION_SAMPLES:-outputs/nchlt-afrikaans/diffusion-50k-retry-20260920/beam-sample-10.txt}" \
 --output-dir "outputs/nchlt-afrikaans/baseline-comparison-${SLURM_JOB_ID}"
