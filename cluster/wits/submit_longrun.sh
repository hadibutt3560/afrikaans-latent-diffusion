#!/usr/bin/env bash
# Explicitly invoked submission only. Never called by the Python training code.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LONGRUN_ENV:-$ROOT/cluster/wits/longrun.env}"
ACTION="${1:?Usage: submit_longrun.sh ACTION HOURS [PREVIOUS_JOB_ID]}"
HOURS="${2:?Provide verified allocation hours, e.g. 2 or 24}"
case "$ACTION" in prepare|smoke|benchmark|run|evaluate|final-test|score) ;; *) echo "Unknown action" >&2; exit 1;; esac
[[ "$HOURS" =~ ^[0-9]+$ ]] && (( HOURS >= 1 && HOURS <= 72 )) || { echo "Hours must be 1..72" >&2; exit 1; }
[[ "$WORKSPACE" = /* && "$WORKSPACE" != *REPLACE* ]] || { echo "Fill in persistent WORKSPACE in longrun.env" >&2; exit 1; }
[[ "$REPO" = /* && -f "$CONFIG" ]] || { echo "Verify REPO and CONFIG" >&2; exit 1; }
mkdir -p "$WORKSPACE/slurm" "$HF_HOME"
export ACTION RUN_SECONDS=$(( HOURS * 3600 - 900 ))
OPTIONS=(--parsable --account="$ACCOUNT" --partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEMORY"
  --time="$HOURS:00:00" --chdir="$REPO" --output="$WORKSPACE/slurm/$ACTION-%j.log" --export=ALL)
if [[ -n "${3:-}" ]]; then
  [[ "$3" =~ ^[0-9]+$ ]] || { echo "Invalid dependency job ID" >&2; exit 1; }
  OPTIONS+=(--dependency="afterok:$3")
fi
JOB=$(sbatch "${OPTIONS[@]}" "${RESOURCE_ARGS[@]}" "$REPO/cluster/wits/afrikaans_longrun.slurm")
JOB="${JOB%%;*}"
printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$ACTION" "$JOB" >> "$WORKSPACE/submitted-jobs.tsv"
printf '%s\n' "$JOB"
