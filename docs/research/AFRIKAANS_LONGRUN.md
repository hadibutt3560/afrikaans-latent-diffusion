# Afrikaans multi-seed diffusion runs (48–72 hours)

Prepared 2 October 2026. Slurm: account `general`, partition `biggpu`, exclusive
allocation, 72-hour maximum. Node health, GPU availability and the persistent
output path are checked on the cluster before submission.

## Design

Two diffusion capacities, trained **from scratch** with at least three paired
training seeds each, on all 47,767 Afrikaans training passages:

| Setting | Small control | Published-capacity model |
|---|---:|---:|
| Transformer width / depth | 256 / 4 | 768 / 12 |
| Parameters | 8,098,176 | 187,978,112 |
| Dense connections | 1 | 3 |
| Frozen AE | mT5-base checkpoint 56531, best step 30,500 | same |
| Latents | 32 × 64, original L2-normalized representation | same |
| Training objective | velocity MSE, cosine noise, self-conditioning 0.5 | same |
| Microbatch × accumulation | 16 × 4 = effective batch 64 | same |
| Precision | FP32 (works on both recorded GPU families) | same |
| AdamW | LR 2e-4, betas .9/.999, weight decay 1e-6 | same |
| LR schedule | 1,000-update warmup; cosine to 5% of peak | same |
| Gradient clip / EMA | 1.0 / .9999, update every optimizer step | same |
| Planned epochs | benchmark-selected 100–400 | identical horizon |

- The larger architecture is from `scripts/reproduction/diffusion_full.sh`.
  Capacity is compared at equal data exposure, not equal FLOPs; exposure far
  exceeds the earlier 50,000-update, batch-8 run (about 8.37 epochs). The AE is
  unchanged, no test text is used, inference seeds are not training replicates,
  and nothing depends on the decoder-aware CE pilot (its human assessment is
  pending).
- An epoch is one seeded permutation of all training examples, including the
  final partial batch: **747 updates/epoch**, so 100–400 epochs is
  **74,700–298,800 updates per model/seed**. The planner picks 3–12 paired seeds
  from 42 and locks the cosine horizon before the first long job. The initial suite
  targets about 72 useful hours on one sequential GPU, excluding queueing.
- Early stopping: at least 60 epochs, then 12 validation checks (every two epochs)
  without .1% relative improvement; `best.pt` tracks every strict improvement. A
  suite finishing under 48 hours adds complete reserve seed pairs with the same
  settings until 48 hours, finishing each started pair. If all 12 pairs finish
  under 48 hours, the runner reports `under_budget` and exits nonzero.

## Data and evaluation

- Frozen-AE latents are encoded once in FP32 in restartable 1,024-example shards;
  `prepare` records truncation at 128 tokens. Data, AE, tokenizer/model files,
  dependency versions and source code are hashed; any change needs a new
  workspace, and source is not edited under a queued or running job.
- Splits: 47,767 / 5,905 / 5,692 passages in 2,005 / 251 / 251 documents; zero
  exact overlap of normalised text, segment ID, document ID and source filename.
  Sampled shared 12-word shingles: 418 / 899 / 184 (train–validation /
  train–test / validation–test), an advisory signal, not an exhaustive leakage
  check. AE test results were reported earlier, so the test set is reserved from
  this tuning but not pristine.
- Every two epochs: EMA velocity MSE on a fixed seeded 2,048-example validation
  subset with three fixed noise draws, and on 512 training examples for the
  generalisation gap. Every ten epochs: 30 matched, unedited DDPM-250 generations
  with the beam-4 decoder (maximum 128, minimum 5, repetition penalty 1.2,
  no-repeat trigrams 3). Self-condition normalisation stays `false`. Test results
  never select checkpoints.
- `evaluate` scores each selected checkpoint on all validation passages, saves
  1,000 generations per run and seals `selection.json` with checkpoint hashes.
  `final-test` evaluates all 5,692 test passages and saves 1,000 generations per
  run with separate seeds. `score` applies the pinned mGPT revision to saved
  generations and matched real-text references and writes blinded
  grammar/coherence review sheets. Every seed is reported.
- Metrics: velocity MSE, external LM perplexity, distinct 1/2/3-grams,
  repetition, EOS, length cap, sentinels, empty outputs (scored as failures) and
  exact training-text matches. They do not establish fluent or correct Afrikaans;
  human review is also needed.
- Logged: loss, unclipped gradient norm, LR, examples/sec, peak allocated/reserved
  GPU memory. Non-finite values stop the run at the last valid checkpoint.
  Summaries: seed means and standard deviations, CSV/JSON, Markdown and SVG curves.

## Environment

Check the working `ld4lg-blackwell` environment without training:

```bash
apptainer exec "$SIF" "$PY" -m pip check
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" "$PY" -c 'import torch, transformers, datasets, einops, accelerate, ema_pytorch, nltk, matplotlib; from tools.longrun_training import make_model; print(torch.__version__, transformers.__version__)'
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" "$PY" -m unittest discover -s tests -p 'test_longrun_common.py' -v
```

Optional separate environment inside the existing Anaconda container:

```bash
apptainer exec "$SIF" conda --version
apptainer exec "$SIF" conda create -y -p "$HOME/.conda/envs/ld4lg-longrun" python=3.10 pip=24.0
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" "$HOME/.conda/envs/ld4lg-longrun/bin/python" -m pip install -r cluster/wits/longrun-requirements.txt
apptainer exec "$SIF" "$HOME/.conda/envs/ld4lg-longrun/bin/python" -m pip check
# Set PY in longrun.env to $HOME/.conda/envs/ld4lg-longrun/bin/python, then:
source cluster/wits/longrun.env
```

The requirements keep Torch 2.7.1/cu128 for Blackwell (not the reproduction
environment's Torch 2.2.2). The smoke test, not the install, confirms node
compatibility. No English spaCy download is needed. Cache the scorer on a
networked login or transfer node:

```bash
apptainer exec --bind "$HOME:$HOME,$HF_HOME:$HF_HOME" "$SIF" "$PY" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("ai-forever/mGPT", revision="40897bd7c8b47a76802c411108ca6220438b8b40"))'
```

## Prepare, smoke-test and benchmark

Short checks, run from `$REPO`:

```bash
P=$(bash cluster/wits/submit_longrun.sh prepare 2)
echo "prepare=$P"
squeue -j "$P"
tail -f "$WORKSPACE/slurm/prepare-$P.log"
```

`cache/manifest.json` must contain `"complete": true`; after a clean pause, rerun
`prepare 2` (completed shards are reused). Inspect `integrity.json` and truncation
fractions before training.

```bash
S=$(bash cluster/wits/submit_longrun.sh smoke 1)
echo "smoke=$S"
tail -f "$WORKSPACE/slurm/smoke-$S.log"
```

Smoke runs both real architectures on real cached data: three updates,
validation, best/latest saves, re-creation, two resumed updates, an exact
model/EMA comparison with an uninterrupted five-update path, and two generations.
`smoke.json` must say `passed`. Rename `$WORKSPACE/smoke` before a retry.

```bash
B=$(bash cluster/wits/submit_longrun.sh benchmark 2)
echo "benchmark=$B"
tail -f "$WORKSPACE/slurm/benchmark-$B.log"
```

Benchmark uses the full-run microbatch and accumulation, discards 30 warmup
updates, measures 200, and times validation, checkpoint writing and 30
generations with CUDA synchronised, saving median, mean, p90, throughput and peak
memory. Partial benchmarks are rejected; rename `benchmark` to retry. Benchmark
outputs never initialise experiment seeds. If memory is short, lower
`micro_batch` and raise `accumulation` to keep batch 64 (e.g. 8 × 8).
`precision: "bf16"` needs a new smoke, benchmark and workspace. FP32 is the
conservative baseline for the recorded Turing/Blackwell inventory.

## Readiness job and locked plan

Alternatively, `cluster/wits/longrun_readiness.slurm` runs smoke, benchmark and
planning in one two-hour allocation on one GPU, stopping at any failed stage and
never starting long training:

```bash
source cluster/wits/longrun.env
R=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --cpus-per-task="$CPUS" --mem="$MEMORY" "${RESOURCE_ARGS[@]}" \
  --chdir="$REPO" --output="$WORKSPACE/slurm/readiness-%j.log" \
  cluster/wits/longrun_readiness.slurm)
printf '%s\n' "$R" | tee "$WORKSPACE/readiness-job-id.txt"
```

On success `plan.json` exists; after a failure, run the remaining individual
action. Standalone planning is CPU-only:

```bash
apptainer exec --bind "$REPO:$REPO,$WORKSPACE:$WORKSPACE" --pwd "$REPO" "$SIF" "$PY" tools/afrikaans_longrun.py plan --config "$CONFIG" --workspace "$WORKSPACE"
cat "$WORKSPACE/benchmark.json"
cat "$WORKSPACE/plan.json"
```

Epoch time for architecture `m`:

```text
747 × measured median update seconds
+ measured validation seconds / 2
+ measured 30-generation seconds / 10
+ measured checkpoint seconds × expected periodic saves per epoch
```

The planner searches 100–400 epochs and 3–12 pairs towards 72 hours and refuses
estimates below 48 or above 79.2 hours. Final generation is estimated separately;
mGPT scoring and queueing are extra. No Wits runtime has been measured yet. A wide
p90/median spread calls for a longer benchmark. If three pairs at 100 epochs are
too slow, a lower `min_epochs` (e.g. 50, six times the earlier exposure) and
early-stop minimum are benchmarked in a new workspace; if 12 pairs at 400 epochs
are too fast, convergence is inspected first. `target_hours` (48–72) is set before
benchmarking. Epochs and schedule never change mid-run.

## Submit

Resumable 24-hour chunks, each leaving 15 minutes for shutdown and skipping
completed model/seed runs:

```bash
J1=$(bash cluster/wits/submit_longrun.sh run 24)
J2=$(bash cluster/wits/submit_longrun.sh run 24 "$J1")
J3=$(bash cluster/wits/submit_longrun.sh run 24 "$J2")
J4=$(bash cluster/wits/submit_longrun.sh run 24 "$J3")
printf 'Training jobs: %s %s %s %s\n' "$J1" "$J2" "$J3" "$J4"
```

Four chunks cover a roughly 72-hour suite plus margins; the last exits once the
suite is complete. `afterok` dependencies stop the chain on numerical errors or OOM; clean timed
pauses exit successfully so the next chunk resumes. A TIMEOUT/NODE_FAIL may block
downstream jobs, which are then resubmitted by hand. Single allocation, if
permitted:

```bash
J=$(bash cluster/wits/submit_longrun.sh run 72)
echo "$J"
# If it reaches its time margin, resume the remaining work in another job.
```

Under a 12-hour limit, use `12` instead of `24` and queue more chunks. Jobs
survive SSH logout; the Python code never calls sbatch.

## Monitor, stop and resume

```bash
squeue -u "$USER"
sacct -j "$J1,$J2,$J3,$J4" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode,NodeList
tail -n 30 "$WORKSPACE"/slurm/run-*.log
cat "$WORKSPACE/suite-status.json"
find "$WORKSPACE/runs" -name status.json -print -exec cat {} \;
apptainer exec --bind "$REPO:$REPO,$WORKSPACE:$WORKSPACE" --pwd "$REPO" "$SIF" "$PY" tools/afrikaans_longrun.py summarize --workspace "$WORKSPACE"
```

A widening train–validation gap with falling train loss signals overfitting.
Persistent sentinels, empty or very short generations, or repetition need
inspection. For a clean pause, cancel unwanted queued continuations first, then:

```bash
scancel --signal=USR1 --batch RUNNING_JOB_ID
# On resume, use the same code, config, inputs and WORKSPACE:
R=$(bash cluster/wits/submit_longrun.sh run 24)
echo "$R"
```

`latest.pt` holds weights, EMA, AdamW moments, RNG states, the epoch/update
cursor, the locked LR schedule and useful time; `best.pt` holds the best
validation EMA. Saves are atomic (temporary file, fsync, replace). Latest is saved
every 250 steps or 10 minutes, before periodic validation and at a graceful stop;
the shell forwards Slurm's 10-minute advance signal
([sbatch documentation](https://slurm.schedmd.com/sbatch.html)) and waits. A node
failure loses only work since the last committed checkpoint. Summaries ignore
uncommitted or replayed log rows. A workspace lock blocks concurrent jobs.
Reproducibility is scoped to the locked hardware/software protocol
([PyTorch guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)).

## Final validation, test and scoring

Once `suite-status.json` reports `completed` (or an accepted `under_budget`) and
no continuation job is queued:

```bash
E=$(bash cluster/wits/submit_longrun.sh evaluate 12)
echo "$E"
# After evaluate completes and selection.json exists:
T=$(bash cluster/wits/submit_longrun.sh final-test 12)
echo "$T"
# After final-test completes:
Q=$(bash cluster/wits/submit_longrun.sh score 12)
echo "$Q"
```

Hours come from `plan.json` within the 72-hour limit. Reruns skip completed
outputs and restart interrupted runs with the same seeds. Each run needs
`selected-validation.json`, `final-test.json` and `test-mgpt.json`; check `status`
and `empty` in the mGPT reports, then:

```bash
apptainer exec --bind "$REPO:$REPO,$WORKSPACE:$WORKSPACE" --pwd "$REPO" "$SIF" "$PY" tools/afrikaans_longrun.py summarize --workspace "$WORKSPACE"
tar -czf "$WORKSPACE/results-without-weights.tar.gz" --exclude='*.pt' --exclude=cache --exclude=slurm --exclude='*.tar.gz' -C "$WORKSPACE" .
# Archive selected weights separately if they must be moved out of purgeable storage:
find "$WORKSPACE/runs" -name best.pt -print
```

Windows download (use the exact `WORKSPACE` path):

```powershell
New-Item -ItemType Directory -Force outputs\cluster-results\longrun-20261002 | Out-Null
scp -o ServerAliveInterval=60 "USER@CLUSTER_HOST:REPLACE_WITH_ABSOLUTE_WORKSPACE/results-without-weights.tar.gz" outputs/cluster-results/longrun-20261002/
```

Reviewers get `blind-review.csv` without `review-key.csv`. Reports give all seeds
with mean/std and the historical AE/test limitation.

## Hardware, storage and files

One CUDA GPU (recorded 48/96-GB class), 4 CPU cores and 64 GB RAM per job; no
multi-GPU speed-up. Persistent space: at least 100 GB for up to 24 runs, or 150 GB
including smoke/benchmark outputs and transferred assets. Train/validation FP32
latents take about 0.44 GB. AE, mT5 and mGPT weights are external; checkpoints are
not in Git.

| Path under WORKSPACE | Contents |
|---|---|
| `integrity.json`, `inputs.json`, `invocations/` | dataset/source hashes, packages, hardware, invocations |
| `cache/manifest.json`, `cache/*.pt` | restartable frozen train/validation latents and truncation audit |
| `smoke.json`, `benchmark.json`, `plan.json` | readiness gates, actual timing, locked epochs and seeds |
| `runs/<model>-seed<N>/config.json` | complete scientific run contract |
| `runs/<model>-seed<N>/latest.pt` | resumable training state |
| `runs/<model>-seed<N>/best.pt` | validation-selected EMA; evaluation only |
| `runs/<model>-seed<N>/metrics.jsonl`, `status.json` | persistent learning curves and committed progress |
| `runs/<model>-seed<N>/*samples*.json`, `*mgpt.json` | unedited generations and pinned scores |
| `slurm/`, `submitted-jobs.tsv` | scheduler logs and submitted IDs |
| `results.json`, `results.csv`, `RESULTS.md`, `training-curves.svg/png` | final/progress summary and plots |
| `selection.json`, `blind-review.csv`, `review-key.csv` | sealed model selection and human assessment |

## Local readiness checks (2 October 2026)

Passed: split integrity audit; syntax/compilation; eight protocol tests; seven
optimizer/checkpoint tests, including exact interrupted-versus-uninterrupted CPU
state (`unittest`, as `pytest` is absent); shell syntax. A launcher test with
mocked container, GPU and scheduler confirmed signal forwarding, waiting for the
checkpoint, exit 0 on a clean pause and exit 23 kept on child failure. Summaries
were generated locally; plots were not, as Matplotlib is absent locally
(`plot-status.json` reports this case). Both architectures passed a synthetic
train/validation/save/resume/sample check on a 12-GB RTX 3060 (Torch 2.11/cu128;
`outputs/longrun-local-checks/model-check.json`), which is not a timing, real-data,
cluster or language-quality result. Cluster `prepare`, `smoke` and `benchmark`
remain required before the epoch/seed count is fixed.
