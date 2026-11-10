#!/usr/bin/env bash
#SBATCH --job-name=ae-diffusion-train
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster111
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source job/common.sh
set +e
# Useful-work budget: the 72 h allocation minus 30 minutes for setup and export.
run_child "$PY" -u job/diffusion_stage.py train --output "$OUT" --model-cache "$MODEL_CACHE" --old-ae "$OLD_AE" --mgpt "$MGPT" --seconds 257400
status=$?
set -e
export_results diffusion-train
printf 'DIFFUSION_TRAIN_EXIT=%s\n' "$status"
date -Is
# A clean stop (75) is reported as success so the finish job, which resumes training, still runs.
if [ "$status" = 75 ]; then exit 0; fi
exit "$status"
