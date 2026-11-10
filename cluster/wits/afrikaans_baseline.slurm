#!/usr/bin/env bash
#SBATCH --job-name=af-ar-baseline
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --exclude=mscluster107,mscluster108,mscluster111

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
PY="$HOME/.conda/envs/ld4lg-blackwell/bin/python"
MATCH="${MATCH:-outputs/nchlt-afrikaans/diffusion-50k-retry-20260920}"
OUT="outputs/nchlt-afrikaans/ar-baseline-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
hostname
date -u
nvidia-smi
test -f "$SIF"
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -c \
 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name()); assert "Blackwell" in torch.cuda.get_device_name(), "Use comparable Blackwell hardware for this budget"; print(torch.ones(8,device="cuda").square().sum().item())'
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -u \
 tools/train_afrikaans_baseline.py --match-run "$MATCH" --output-dir "$OUT" \
 --budget-seconds 16266 --save-every 250 --validation-examples 128 \
 --sample-count 100 --sample-max-tokens 64
echo "BASELINE_OUTPUT=$OUT"
date -u
