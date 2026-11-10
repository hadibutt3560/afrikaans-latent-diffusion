#!/usr/bin/env bash
# Fixed-weight confirmation study; no training or checkpoint modification.
set -euo pipefail
cd /mnt/c/Users/SCIENTIST/Documents/GitHub/research
export HF_HOME=/home/scientist/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=/home/scientist/miniconda3/envs/ld4lg-reproduction/bin/python
ROOT="outputs/nchlt-afrikaans/combined-inference-$(date +%Y%m%d-%H%M%S)"
mkdir "$ROOT"
exec > >(tee -a "$ROOT/console.log") 2>&1
trap 'echo FAILED > "$ROOT/STATUS.txt"' ERR
echo RUNNING > "$ROOT/STATUS.txt"
echo "RESULTS=$ROOT"
date -u
for STEPS in 50 250; do
    for ARM in ddim64 ddim128 ddpm128; do
        SAMPLER=ddim
        LENGTH=128
        if [ "$ARM" = ddim64 ]; then LENGTH=64; fi
        if [ "$ARM" = ddpm128 ]; then SAMPLER=ddpm; fi
        echo "Generating $ARM at $STEPS steps"
        "$PY" -u tools/trace_afrikaans_inference.py \
            --diffusion-dir outputs/cluster-results/diffusion-50k-retry-20260920 \
            --autoencoder-dir outputs/cluster-results/afrikaans-56531/best \
            --output-dir "$ROOT/$ARM-$STEPS" \
            --sampler "$SAMPLER" --decode-max-length "$LENGTH" \
            --steps "$STEPS" --seeds $(seq 82 101) --every 1 --decode-batch-size 4 \
            > "$ROOT/$ARM-$STEPS.log" 2>&1
        echo "Scoring $ARM at $STEPS steps"
        "$PY" -u tools/analyze_inference_trace.py \
            --trace-dir "$ROOT/$ARM-$STEPS" \
            --output-dir "$ROOT/quality-$ARM-$STEPS" \
            --scorer-snapshot "$HF_HOME/hub/models--ai-forever--mGPT/snapshots/40897bd7c8b47a76802c411108ca6220438b8b40" \
            > "$ROOT/quality-$ARM-$STEPS.log" 2>&1
        echo "Completed $ARM at $STEPS steps"
    done
done
echo COMPLETED > "$ROOT/STATUS.txt"
date -u
echo "All six conditions generated and scored: $ROOT"
