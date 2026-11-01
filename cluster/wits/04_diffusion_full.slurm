#!/usr/bin/env bash
#SBATCH --job-name=ld4lg-diff-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
##SBATCH --account=REPLACE_WITH_WITS_ACCOUNT
##SBATCH --partition=REPLACE_WITH_WITS_GPU_PARTITION

set -euo pipefail
: "${LATENT_MODEL_PATH:?Submit with --export=ALL,LATENT_MODEL_PATH=/absolute/autoencoder/output}"
cd "${RESEARCH_ROOT:-${SLURM_SUBMIT_DIR}}"
source cluster/wits/common_env.sh
export OUTPUT_DIR="${DIFFUSION_OUTPUT_DIR:-${LD4LG_SCRATCH_ROOT}/outputs/roc-diff-full-${SLURM_JOB_ID}}"
bash scripts/reproduction/diffusion_full.sh
