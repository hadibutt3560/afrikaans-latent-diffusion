#!/usr/bin/env bash
#SBATCH --job-name=ae-diffusion-finish
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster111
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source job/common.sh
set +e
run_child "$PY" -u job/diffusion_stage.py finish --output "$OUT" --model-cache "$MODEL_CACHE" --old-ae "$OLD_AE" --mgpt "$MGPT" --seconds 127800
status=$?
set -e
export_results final
printf 'DIFFUSION_FINISH_EXIT=%s\n' "$status"
date -Is
exit "$status"
