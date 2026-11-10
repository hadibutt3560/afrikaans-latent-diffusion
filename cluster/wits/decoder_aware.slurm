#!/usr/bin/env bash
#SBATCH --job-name=af-decoder-ce
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --exclude=mscluster107,mscluster108,mscluster111

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
SIF="${SIF:-$HOME/containers/anaconda-2025-12-2.sif}"
PY="${PY:-$HOME/.conda/envs/ld4lg-blackwell/bin/python}"
BASE="${BASE:-$HOME/research-afrikaans-24h}"
OUT="${OUT:-outputs/decoder-aware-${SLURM_JOB_ID}}"
RESUME_ARGS=()
if [ "${RESUME:-0}" = 1 ]; then RESUME_ARGS+=(--resume); fi
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
test -f "$SIF"
test -x "$PY"
hostname
date -u
nvidia-smi
sha256sum -c bundle-manifest.sha256
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -u tools/decoder_cycle_preflight.py --base "$BASE"
# Two matched fine-tuning arms, not an unrestricted 24/72-hour job.
apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" -u tools/decoder_aware_cycle.py \
  --diffusion-dir "$BASE/outputs/nchlt-afrikaans/diffusion-50k-retry-20260920" \
  --autoencoder-dir "$BASE/outputs/afrikaans-24h-56531/best" \
  --model-cache "$BASE/models/mt5-base" \
  --dataset-dir data/nchlt_afrikaans/processed/stage_5_7 \
  --output-dir "$OUT" --steps 600 --arm-seconds 1200 \
  --train-examples 4096 --valid-examples 64 --accumulation 8 \
  --save-every 25 --sample-count 30 --sample-start 150 "${RESUME_ARGS[@]}" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM INT
wait "$CHILD"
echo "CYCLE_OUTPUT=$OUT"
date -u
