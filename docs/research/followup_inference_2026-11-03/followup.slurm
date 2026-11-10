#!/usr/bin/env bash
#SBATCH --job-name=ae-followup
#SBATCH --account=general
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster[108,110,112]
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source job/common.sh
set +e
run_child "$PY" -u job/followup.py --output "$PWD/outputs/followup" --model-cache "$MODEL_CACHE" --old-ae "$OLD_AE" \
  --mgpt "$MGPT" --b0-runs "$PWD/outputs/b0-runs" --protocol job/FOLLOWUP_PROTOCOL.md --seconds 70200
status=$?
set -e
export_results followup
printf 'FOLLOWUP_EXIT=%s\n' "$status"
date -Is
exit "$status"
