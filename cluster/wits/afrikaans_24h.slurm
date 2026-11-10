#!/usr/bin/env bash
#SBATCH --job-name=afrikaans-ae
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=1-00:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted bundle directory}"
hostname
date -u
nvidia-smi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
SIF=/usr/local/containers/anaconda/anaconda-2025-12-2.sif
PY="$HOME/.conda/envs/ld4lg-blackwell/bin/python"
test -f "$SIF"
test -x "$PY"
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -u tools/wits_afrikaans_train.py
