#!/usr/bin/env bash
#SBATCH --job-name=ld4lg-diff-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
##SBATCH --account=REPLACE_WITH_WITS_ACCOUNT
##SBATCH --partition=REPLACE_WITH_WITS_GPU_PARTITION

set -euo pipefail
: "${LATENT_MODEL_PATH:?Submit with --export=ALL,LATENT_MODEL_PATH=/absolute/autoencoder/output}"
cd "${RESEARCH_ROOT:-${SLURM_SUBMIT_DIR}}"
source cluster/wits/common_env.sh
export OUTPUT_DIR="${LD4LG_SCRATCH_ROOT}/outputs/roc-diff-smoke-${SLURM_JOB_ID}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export RUN_RESUME_TEST=1
bash scripts/reproduction/diffusion_smoke.sh
