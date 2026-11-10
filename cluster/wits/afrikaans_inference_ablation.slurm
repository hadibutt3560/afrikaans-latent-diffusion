#!/usr/bin/env bash
#SBATCH --job-name=af-inference-fix
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
DIFF=outputs/nchlt-afrikaans/diffusion-50k-retry-20260920
AE=outputs/afrikaans-24h-56531/best
OUT="outputs/nchlt-afrikaans/inference-ablation-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
test -f "$SIF"
test -f "$DIFF/model.pt"
test -f "$AE/model.pt"
mkdir -p "$OUT"
hostname
date -u
nvidia-smi
run_python() {
    apptainer exec --nv --bind "$HOME:$HOME" --pwd "$PWD" "$SIF" "$PY" "$@"
}
run_python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name()); print(torch.ones(8,device="cuda").square().sum().item())'
sha256sum "$DIFF/model.pt" "$AE/model.pt" > "$OUT/checkpoint-hashes.txt"
# Fresh seeds; all arms use identical seeds and only one factor changes from control.
for STEPS in 50 250; do
    for ARM in control ddim projection length128; do
        EXTRA=()
        case "$ARM" in
            ddim) EXTRA+=(--sampler ddim) ;;
            projection) EXTRA+=(--project-decoder-latents) ;;
            length128) EXTRA+=(--decode-max-length 128) ;;
        esac
        run_python -u tools/trace_afrikaans_inference.py \
            --diffusion-dir "$DIFF" --autoencoder-dir "$AE" \
            --output-dir "$OUT/${ARM}-${STEPS}" --steps "$STEPS" \
            --seeds $(seq 62 81) --every 1 --decode-batch-size 4 \
            "${EXTRA[@]}" > "$OUT/${ARM}-${STEPS}.log" 2>&1
        echo "Completed $ARM at $STEPS steps"
    done
done
echo "ABLATION_OUTPUT=$OUT"
date -u
