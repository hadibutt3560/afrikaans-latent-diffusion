#!/usr/bin/env bash
#SBATCH --job-name=ld4lg-ae-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --account=REPLACE_WITH_WITS_ACCOUNT
##SBATCH --partition=REPLACE_WITH_WITS_GPU_PARTITION

set -euo pipefail
cd "${RESEARCH_ROOT:-${SLURM_SUBMIT_DIR}}"
source cluster/wits/common_env.sh
export OUTPUT_DIR="${AE_OUTPUT_DIR:-${LD4LG_SCRATCH_ROOT}/outputs/roc-ae-full-${SLURM_JOB_ID}}"
bash scripts/reproduction/autoencoder_full.sh
