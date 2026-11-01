# Ready-to-run 48–72-hour Afrikaans experiment

Prepared 2 October 2026. **No cluster jobs have been submitted.** The author
confirmed Slurm account `general`, partition `biggpu`, exclusive allocation,
and a 72-hour maximum. Current healthy nodes, GPU availability, and the approved
persistent output path still have to be checked on the cluster.

## Experiment and what it can answer

Train two diffusion capacities **from scratch**, each with at least three
independent, paired training seeds, over all 47,767 Afrikaans training passages:

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

The larger architecture comes from `scripts/reproduction/diffusion_full.sh`.
This compares capacity at equal data exposure, not equal FLOPs. It increases
exposure substantially over the earlier 50,000-update, batch-8 run (about 8.37
epochs). It does not assume the decoder-aware CE pilot succeeded, alter the
frozen AE, mix in test text, or treat inference seeds as training replicates.
The previous decoder-aware pilot's human quality assessment remains pending.

An epoch visits **every training example once**, with a new seeded permutation,
including the final partial batch. At batch 64 there are **747 updates/epoch**:
100–400 epochs means **74,700–298,800 updates per model/seed**. The timing planner
chooses 3–12 paired training seeds beginning at 42 and locks the entire cosine
horizon before the first long job. The initial suite targets approximately
72 useful hours on one sequential GPU; this is elapsed allocated compute time,
not 72 hours multiplied by concurrent GPUs, and excludes queue waiting.

Early stopping requires at least 60 epochs and 12 validation checks without a
relative improvement of .1% (checks every two epochs). `best.pt` still tracks
every strict validation improvement. If the initial suite finishes below 48
hours, the runner starts complete reserve seed pairs with the same locked
settings until it reaches 48 hours. It finishes each started pair. If all 12
pairs finish below 48 hours, it reports `under_budget` and exits nonzero; it
does not waste time or claim the requested duration was achieved. In that case,
review the observed learning curves and design further justified experiments.
Actual runtime may exceed the estimate; clean continuation is supported.

## Inputs, integrity, validation, and final evaluation

The frozen autoencoder's latents are encoded once in FP32, in restartable
1,024-example shards. Training uses this complete cache, avoiding repeated
frozen-encoder work. `prepare` also records truncation rates at 128 tokens.
Data, AE, tokenizer/model files, dependency versions, and relevant source code
are hashed. Changed inputs or code require a new workspace. Do not edit the
source under a queued/running experiment.

Local integrity checks found 47,767 / 5,905 / 5,692 train/validation/test passages
in 2,005 / 251 / 251 documents. Exact normalized text, segment ID, document ID,
and source filename overlaps are zero. Sampled shared 12-word shingles number
418 / 899 / 184 for train–validation / train–test / validation–test. These are
advisory boilerplate/near-duplicate signals, not an exhaustive semantic leakage
check. No test metrics are used to resolve them or tune the run. The project has
already reported AE test results historically, so the test set is reserved from
this new diffusion tuning, not claimed pristine across the project's history.

Every two epochs, evaluate EMA velocity MSE on a fixed seeded 2,048-example
validation subset, using three independent, fixed noise draws. The same checks
on 512 training examples show the generalization gap. Validation saves the best
EMA; training saves resumable latest state. Every ten epochs, save 30 matched,
unedited generations with DDPM-250 and the existing beam-4 decoder (maximum 128,
minimum 5, repetition penalty 1.2, no-repeat trigrams 3). Do not select checkpoints
using test results. Sampling self-condition normalization stays at the existing
default `false`; that earlier inference ablation is not silently introduced.

After training, `evaluate` evaluates each selected checkpoint on **all validation
passages**, saves 1,000 new generations per run, and seals `selection.json` with
checkpoint hashes. `final-test` then evaluates all 5,692 test passages and saves
1,000 generations with separate seeds per run. Each training seed is reported;
no best-seed cherry-picking. `score` uses the project's pinned mGPT revision on
saved generations and matched real-text references. It also writes blinded
grammar/coherence review sheets. Automatic metrics include velocity MSE,
external LM perplexity, distinct 1/2/3-grams, repetition, EOS, length cap,
sentinels, empty outputs, and exact training-text matches. Empty outputs are
explicit scoring failures, not silently dropped. These diagnostics do not prove
fluent or factually correct Afrikaans; use fluent-human review as well.

Loss, unclipped gradient norm, LR, examples/sec, and peak allocated/reserved GPU
memory are logged. Non-finite loss/gradients/parameters stop the run and retain
the last valid checkpoint. Summary artifacts include seed means/standard
deviations, CSV/JSON results, a Markdown report and standalone SVG curves.

## Environment setup

Prefer the working `ld4lg-blackwell` environment. Check it without training:

```bash
apptainer exec "$SIF" "$PY" -m pip check
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" "$PY" -c 'import torch, transformers, datasets, einops, accelerate, ema_pytorch, nltk, matplotlib; from tools.longrun_training import make_model; print(torch.__version__, transformers.__version__)'
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" "$PY" -m unittest discover -s tests -p 'test_longrun_common.py' -v
```

If a separate environment is needed, create it without changing the working
one. The container is an existing Anaconda image; verify `conda` is available:

```bash
apptainer exec "$SIF" conda --version
apptainer exec "$SIF" conda create -y -p "$HOME/.conda/envs/ld4lg-longrun" python=3.10 pip=24.0
apptainer exec --bind "$REPO:$REPO" --pwd "$REPO" "$SIF" "$HOME/.conda/envs/ld4lg-longrun/bin/python" -m pip install -r cluster/wits/longrun-requirements.txt
apptainer exec "$SIF" "$HOME/.conda/envs/ld4lg-longrun/bin/python" -m pip check
# Set PY in longrun.env to $HOME/.conda/envs/ld4lg-longrun/bin/python, then:
source cluster/wits/longrun.env
```

The requirements retain the demonstrated Torch 2.7.1/cu128 stack for Blackwell.
Do not install the older reproduction environment's Torch 2.2.2 on Blackwell.
Package compatibility on the actual node is verified by the smoke, not assumed
from successful installation. No English spaCy download is required here.

Cache the external scorer on an approved networked/login or transfer node:

```bash
apptainer exec --bind "$HOME:$HOME,$HF_HOME:$HF_HOME" "$SIF" "$PY" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("ai-forever/mGPT", revision="40897bd7c8b47a76802c411108ca6220438b8b40"))'
```

## Prepare, smoke-test and benchmark on the intended GPU

These commands submit **short checks only**. Run them after reviewing the
configuration. From `$REPO`:

```bash
P=$(bash cluster/wits/submit_longrun.sh prepare 2)
echo "prepare=$P"
squeue -j "$P"
tail -f "$WORKSPACE/slurm/prepare-$P.log"
```

After completion, require `cache/manifest.json` to contain `"complete": true`.
If it cleanly pauses first, rerun the same `prepare 2` command; completed shards
are reused. Inspect `integrity.json` and cache truncation fractions before training.

```bash
S=$(bash cluster/wits/submit_longrun.sh smoke 1)
echo "smoke=$S"
tail -f "$WORKSPACE/slurm/smoke-$S.log"
```

Smoke uses **both real architectures and real cached data**, trains three updates,
validates, saves best/latest, recreates the model/optimizer, resumes two updates,
and compares its model/EMA exactly against an uninterrupted five-update path.
It generates two texts through the frozen AE. `smoke.json` must say `passed`.
Expect meaningless language after five updates; this is a pipeline check.
Smoke deliberately refuses to overwrite an existing attempt. Preserve a failed
attempt by renaming `$WORKSPACE/smoke` before retrying after fixing the cause.

```bash
B=$(bash cluster/wits/submit_longrun.sh benchmark 2)
echo "benchmark=$B"
tail -f "$WORKSPACE/slurm/benchmark-$B.log"
```

Benchmark trains each architecture with the **full-run microbatch and accumulation**,
discards 30 warmup updates, measures 200 updates, and times the real validation,
checkpoint write, and 30-generation operations. CUDA is synchronized for timing;
median, mean, p90, throughput and peak GPU allocation are saved. A partial
benchmark is rejected. Preserve/rename the `benchmark` directory to retry one.
Benchmark outputs are separate from experiment seeds and never initialize them.

If memory is insufficient, lower `micro_batch` and raise `accumulation` to keep
batch 64 (e.g. 8 × 8). If all intended nodes support BF16, `precision: "bf16"`
is available, but must pass a new smoke and benchmark. Use a new workspace after
any protocol change. FP32 is the conservative common baseline for the recorded
Turing/Blackwell inventory, not a claim that it is fastest.

## Calculate the locked 72-hour plan

Alternatively, after preparation completes, `cluster/wits/longrun_readiness.slurm`
runs smoke, benchmark and planning in **one two-hour allocation**, on the same GPU.
This avoids separate queue waits. It preserves the hashed Python code and cache
contract, stops if a stage fails, and does not submit or execute long training.
Use it for a fresh readiness attempt, before separate smoke/benchmark jobs:

```bash
source cluster/wits/longrun.env
R=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --cpus-per-task="$CPUS" --mem="$MEMORY" "${RESOURCE_ARGS[@]}" \
  --chdir="$REPO" --output="$WORKSPACE/slurm/readiness-%j.log" \
  cluster/wits/longrun_readiness.slurm)
printf '%s\n' "$R" | tee "$WORKSPACE/readiness-job-id.txt"
```

If it succeeds, `plan.json` already exists and the following standalone planning
command is unnecessary. If it fails after smoke, inspect the error and use the
individual remaining action; completed readiness evidence is not overwritten.

This command is CPU-only and may run on the login node:

```bash
apptainer exec --bind "$REPO:$REPO,$WORKSPACE:$WORKSPACE" --pwd "$REPO" "$SIF" "$PY" tools/afrikaans_longrun.py plan --config "$CONFIG" --workspace "$WORKSPACE"
cat "$WORKSPACE/benchmark.json"
cat "$WORKSPACE/plan.json"
```

For architecture `m`, the planner estimates epoch time as:

```text
747 × measured median update seconds
+ measured validation seconds / 2
+ measured 30-generation seconds / 10
+ measured checkpoint seconds × expected periodic saves per epoch
```

It searches 100–400 epochs and 3–12 pairs to approach 72 hours. It refuses an
estimate below 48 hours or above 79.2 hours within those scientific bounds.
Final validation/test generation is estimated separately from benchmark sampling
times and added to the report; mGPT scoring and queues are additional. The
planner does not use historical pilot throughput as a forecast for this larger
model and batch. No numeric hardware runtime has been measured here on Wits.
Check p90/median spread: large variation warrants a longer benchmark or a stable
node before locking the plan. If even three pairs at 100 epochs exceed the
budget, review a lower `min_epochs` (for example 50, still six times the earlier
data exposure) and an appropriate early-stop minimum, then repeat the benchmark
in a new workspace. If 12 pairs at 400 epochs are too fast, inspect convergence
before approving any larger scientific scope. Set `target_hours` to 48–72 before benchmarking
if you prefer a different target. Never change epochs or the schedule mid-run.

## Submit the long experiment — only when you are ready

Recommended: resumable 24-hour chunks, leaving 15 minutes for orderly shutdown.
Each chunk resumes the same sequential suite and skips completed model/seeds.
Run these commands yourself after the smoke and plan are inspected:

```bash
J1=$(bash cluster/wits/submit_longrun.sh run 24)
J2=$(bash cluster/wits/submit_longrun.sh run 24 "$J1")
J3=$(bash cluster/wits/submit_longrun.sh run 24 "$J2")
J4=$(bash cluster/wits/submit_longrun.sh run 24 "$J3")
printf 'Training jobs: %s %s %s %s\n' "$J1" "$J2" "$J3" "$J4"
```

Four chunks accommodate a roughly 72-hour suite plus shutdown/restart margins;
the last job exits once the suite is complete. Dependencies use `afterok`:
numerical errors/OOMs stop the chain; clean timed pauses exit successfully so
the next chunk can resume. A forced TIMEOUT/NODE_FAIL may also block downstream
jobs, so inspect and explicitly resubmit after checking the cause.

If a single long allocation is preferable and permitted:

```bash
J=$(bash cluster/wits/submit_longrun.sh run 72)
echo "$J"
# If it reaches its time margin, resume the remaining work in another job.
```

If policy later allows only 12 hours, replace `24` with `12` and queue enough
chunks. Nothing depends on one uninterrupted allocation. Jobs survive SSH
logout. No automatic sbatch call occurs in the Python code.

## Monitor, stop, and resume

```bash
squeue -u "$USER"
sacct -j "$J1,$J2,$J3,$J4" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode,NodeList
tail -n 30 "$WORKSPACE"/slurm/run-*.log
cat "$WORKSPACE/suite-status.json"
find "$WORKSPACE/runs" -name status.json -print -exec cat {} \;
apptainer exec --bind "$REPO:$REPO,$WORKSPACE:$WORKSPACE" --pwd "$REPO" "$SIF" "$PY" tools/afrikaans_longrun.py summarize --workspace "$WORKSPACE"
```

Watch validation/train gap, LR curves, finite gradient norms, GPU memory and
examples/sec. Increasing train–validation gap with declining train loss is an
overfitting signal; a low denoising loss alone is not a fluency claim. Persistent
sentinels/empty generations, repetition or very short text warrant inspection.

For a deliberate clean pause, signal the running batch shell. **First cancel
queued continuation job IDs if you do not want them to restart work.**

```bash
scancel --signal=USR1 --batch RUNNING_JOB_ID
# On resume, use the same code, config, inputs and WORKSPACE:
R=$(bash cluster/wits/submit_longrun.sh run 24)
echo "$R"
```

`latest.pt` contains weights, EMA, AdamW moments, RNG states, epoch/update cursor,
locked LR schedule and cumulative useful time. Best validation EMA is `best.pt`.
Saves use a same-directory temporary file, fsync and atomic replacement. Latest
is saved every 250 steps or 10 minutes, before periodic validation, and at a
graceful stop. Slurm gives a 10-minute advance signal, which the shell forwards
and waits for. An abrupt node failure loses only work since the last committed
checkpoint; fsync durability still depends on the cluster filesystem. Logs append;
summary ignores uncommitted or replayed log rows. A workspace lock blocks two
simultaneous jobs. Do not launch this Python entry point concurrently outside
the provided launcher.

Signal semantics follow the [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).
Reproducibility is scoped to the locked hardware/software protocol: see
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Collect final validation, test and language-quality results

Only after `suite-status.json` reports `completed` (or explicitly acknowledge
`under_budget` if scientifically finished early):

Let queued continuation jobs finish or cancel the unused ones before starting
evaluation; the workspace lock intentionally prevents concurrent writers.

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

Choose evaluation hours from `plan.json` within the 72-hour limit. Repeating the
same action skips completed per-run outputs; an interrupted incomplete run's
evaluation restarts with the same fixed seeds. Check that every completed
training run has `selected-validation.json`, `final-test.json` and
`test-mgpt.json`. Inspect `status` and `empty` in mGPT reports, then:

```bash
apptainer exec --bind "$REPO:$REPO,$WORKSPACE:$WORKSPACE" --pwd "$REPO" "$SIF" "$PY" tools/afrikaans_longrun.py summarize --workspace "$WORKSPACE"
tar -czf "$WORKSPACE/results-without-weights.tar.gz" --exclude='*.pt' --exclude=cache --exclude=slurm --exclude='*.tar.gz' -C "$WORKSPACE" .
# Archive selected weights separately if they must be moved out of purgeable storage:
find "$WORKSPACE/runs" -name best.pt -print
```

On Windows, replace the persistent cluster path with the exact `WORKSPACE`:

```powershell
New-Item -ItemType Directory -Force outputs\cluster-results\longrun-20261002 | Out-Null
scp -o ServerAliveInterval=60 "USER@CLUSTER_HOST:REPLACE_WITH_ABSOLUTE_WORKSPACE/results-without-weights.tar.gz" outputs/cluster-results/longrun-20261002/
```

Provide `blind-review.csv` without `review-key.csv` to fluent reviewers. Report
all training seeds with mean/std and the historical AE/test limitation. Review
quality before interpreting capacity differences as a research improvement.

## Hardware, storage and artifact map

Use one CUDA GPU per job, 4 CPU cores and 64 GB host RAM. Target the recorded
48/96-GB GPU class; exact peak memory is a smoke/benchmark result. Diffusion
training needs weights, gradients, AdamW state and a separate EMA plus the
frozen mT5 decoder. No distributed training or multi-GPU speedup is assumed.
Reserve **at least 100 GB of approved persistent space** for up to 24 runs,
temporary checkpoint replacement space, logs, generation text and caches;
increase to 150 GB if keeping all smoke/benchmark and transferred model assets.
The train/validation FP32 latent payload alone is about 0.44 GB. AE and mT5/mGPT
weights remain external, existing assets; large checkpoints are not in Git.

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

## Readiness evidence

Local checks: the full split integrity audit; syntax/compilation checks; eight
dependency-free protocol tests; seven optimizer/checkpoint tests including exact
interrupted-versus-uninterrupted CPU state; shell syntax checks. The initial
sandbox could not launch the existing Windows Store Python; the same environment
worked outside that sandbox. `pytest` is absent there, so the new tests use
`unittest` and do not require installing it.

The local launcher check also passed: a simulated advance signal reached its
child, the launcher waited for the child's checkpoint, clean pause became exit
0, and a genuine child failure retained exit 23. Container, GPU and scheduler
commands were mocked for this check; no Slurm submission occurred.

JSON/CSV/Markdown summary generation was exercised locally. Matplotlib is absent
from the existing local environment, so plot rendering remains a cluster check;
the supplied requirements include it. If absent, `plot-status.json` reports the
missing dependency while preserving the tabular results.

Both actual architectures also passed the synthetic GPU train/validation/save/
resume/sample check on a local 12-GB RTX 3060 (Torch 2.11/cu128), recorded in
`outputs/longrun-local-checks/model-check.json`. It is not an
intended-hardware timing measurement, real-data AE test, cluster signal test,
or language-quality result. Cluster `prepare`, `smoke`, and `benchmark` remain
mandatory. The final epoch/seed count is deliberately unresolved until that
benchmark exists. No long job or remote submission has been performed.
