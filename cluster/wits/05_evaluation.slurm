#!/usr/bin/env bash
#SBATCH --job-name=ld4lg-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
##SBATCH --account=REPLACE_WITH_WITS_ACCOUNT
##SBATCH --partition=REPLACE_WITH_WITS_GPU_PARTITION

set -euo pipefail
: "${DIFFUSION_OUTPUT_DIR:?Submit with --export=ALL,DIFFUSION_OUTPUT_DIR=/absolute/diffusion/output}"
cd "${RESEARCH_ROOT:-${SLURM_SUBMIT_DIR}}"
source cluster/wits/common_env.sh

python train_text_diffusion.py \
  --eval_test \
  --resume_dir "${DIFFUSION_OUTPUT_DIR}" \
  --output_dir "${DIFFUSION_OUTPUT_DIR}" \
  --num_samples 1000 \
  --sampling_timesteps 250 \
  --sampler ddpm \
  --sampling_schedule cosine
