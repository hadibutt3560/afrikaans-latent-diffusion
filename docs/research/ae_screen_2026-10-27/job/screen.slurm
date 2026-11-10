#!/usr/bin/env bash
#SBATCH --job-name=ae-screen
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster111
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source job/common.sh
set +e
run_child "$PY" -u job/screen.py --output "$OUT" --old-ae "$OLD_AE" --model-cache "$MODEL_CACHE"
status=$?
set -e
export_results screen
printf 'SCREEN_EXIT=%s\n' "$status"
date -Is
# 75 is a clean stop: resubmit screen.slurm alone and it resumes.
exit "$status"
