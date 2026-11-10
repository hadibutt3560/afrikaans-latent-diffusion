#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/SCIENTIST/Documents/GitHub/research
export HF_HOME=/home/scientist/.cache/huggingface
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
PY=/home/scientist/miniconda3/envs/ld4lg-reproduction/bin/python
SEED_START="${SEED_START:-102}"
SEED_END="${SEED_END:-111}"
OUT="outputs/nchlt-afrikaans/self-condition-check-$(date +%Y%m%d-%H%M%S)"
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
trap 'echo FAILED > "$OUT/STATUS.txt"' ERR
echo RUNNING > "$OUT/STATUS.txt"
echo "RESULTS=$OUT"
echo "Matched seeds: $SEED_START through $SEED_END"
for ARM in control aligned; do
    FLAGS=()
    if [ "$ARM" = aligned ]; then FLAGS+=(--normalize-sampling-self-condition); fi
    echo "Generating $ARM"
    "$PY" -u tools/trace_afrikaans_inference.py \
        --diffusion-dir outputs/cluster-results/diffusion-50k-retry-20260920 \
        --autoencoder-dir outputs/cluster-results/afrikaans-56531/best \
        --output-dir "$OUT/$ARM" --sampler ddim --steps 50 \
        --decode-max-length 128 --seeds $(seq "$SEED_START" "$SEED_END") --every 1 --decode-batch-size 4 \
        "${FLAGS[@]}" > "$OUT/$ARM.log" 2>&1
    echo "Scoring $ARM"
    "$PY" -u tools/analyze_inference_trace.py --trace-dir "$OUT/$ARM" \
        --output-dir "$OUT/quality-$ARM" \
        --scorer-snapshot "$HF_HOME/hub/models--ai-forever--mGPT/snapshots/40897bd7c8b47a76802c411108ca6220438b8b40" \
        > "$OUT/quality-$ARM.log" 2>&1
    echo "Completed $ARM"
done
"$PY" tools/summarize_self_condition.py "$OUT"
echo COMPLETED > "$OUT/STATUS.txt"
