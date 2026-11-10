#!/usr/bin/env bash
#SBATCH --job-name=ld4lg-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
# Add the verified Wits account and GPU partition here:
##SBATCH --account=REPLACE_WITH_WITS_ACCOUNT
##SBATCH --partition=REPLACE_WITH_WITS_GPU_PARTITION

set -euo pipefail
cd "${RESEARCH_ROOT:-${SLURM_SUBMIT_DIR}}"
source cluster/wits/common_env.sh

python tools/preflight.py \
  --require-cuda \
  --require-bf16 \
  --require-model-cache \
  --load-model \
  --output-dir "${LD4LG_SCRATCH_ROOT}/outputs/preflight-${SLURM_JOB_ID}"
