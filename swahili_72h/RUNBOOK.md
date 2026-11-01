# Swahili latent diffusion: one 72-hour allocation

Prepared only. No cluster submission has been made. All files live in the separate `swahili_72h` workspace; the Afrikaans source, outputs and prior uncommitted changes are preserved.

## Experiment

Use Jordan's exact 664/37/37 file split. See `DATA_AUDIT.md`: original document boundaries are absent and there is cross-split overlap. No text is removed. The exported 48-word windows are lossless and have stable IDs; 437,326 / 29,761 / 25,537 passages fit the 192-token mT5 view without truncation. Share these exports with the discrete pipeline to establish passage-level alignment; no communication to Jordan has been sent.

Start from **pretrained multilingual mT5-base**, with a new 32-by-64 normalized Perceiver bottleneck. Train the bottleneck and complete decoder on Swahili reconstruction; freeze encoder, token embeddings and output projection. Do not load the Afrikaans-trained AE. mT5 pretraining includes Swahili, but pretraining does not establish reconstruction quality. Both pretrained weight files match upstream LFS checksums; revisions and hashes are in `assets-provenance.json`. Sources: [mT5 model card](https://huggingface.co/google/mt5-base), [mGPT model card](https://huggingface.co/ai-forever/mGPT).

Select AE by fixed validation token NLL. Before caching, require token accuracy >=35%, mean character chrF >=25, and NLL <=80% of the initialization baseline. These are preregistered engineering gates, not proof of fluency. Failure produces diagnostic reconstructions and a partial archive; do not spend days diffusing unusable latents or silently lower gates.

Compare small (256 width, 4 layers) and published-size (768 width, 12 layers) diffusion transformers. Cosine noise, velocity prediction, L2 loss, EMA .9999, original repository self-conditioning behavior, DDPM250 sampling and beam-4 decoding. The frozen extracted model implementation is in `vendor`; extraction provenance is recorded. These settings are not an exact replication of every Afrikaans condition: language, data, length and AE training differ.

Primary admission order: small-42, published-42, small-43, published-43. Useful follow-ups: the same pair for seeds 44, 45, 46, in that order. Fewer complete runs are admitted if measured costs require it. Validation-only selection: fixed 1,024-example MSE probe with three fixed noise repeats during training; lowest probe-MSE EMA checkpoint per run, then full-validation evaluation and selection of the overall winner. All admitted runs, not only the winner, get full test denoising evaluation and 256 validation-seeded plus 256 independently test-seeded unconditional generations. These generation group labels denote independent RNG banks, not conditioning on held-out text. Actual test text is first encoded/evaluated after `selection.json` freezes checkpoints. There is no test-driven tuning.

External scoring uses pinned mGPT in FP32, context 512/stride 256, scoring each token once with BOS. Score every generated sample plus 256 fixed real references per split. Empty outputs remain visible. Report token-weighted and mean sample perplexity, diversity, lengths, validation/test losses and human-review sheets. Perplexity is exploratory and incomparable across unlike tokenizers/languages; mGPT pretraining overlap is unknown. Human assessment is outstanding until Swahili-speaking reviewers fill the blinded sheet.

Review rubric: grammar 1 = mostly malformed, 5 = well-formed; coherence 1 = unrelated fragments, 5 = a consistent passage; natural Swahili 1 = largely unnatural, 5 = idiomatic. Intermediate scores represent increasing quality. Use at least two fluent reviewers, randomize sheet order, retain independent scores and disagreements, and do not reveal group labels until scoring is complete. No automatic ratings are substituted for humans.

## Stage budget

| Stage | Bound / typical reservation |
|---|---:|
| Offline integrity, GPU/environment, AE/diffusion resume smoke, initial reconstruction and scorer timing | 1 h maximum |
| Swahili AE adaptation | 14 h maximum; early stopping permitted |
| AE reconstruction gate | 0.5 h maximum |
| Train/validation latent cache | 3 h maximum |
| Real-cache diffusion, DDPM250 generation, encoding and validation benchmark | 0.5 h maximum |
| Diffusion training | Remaining measured budget; nominal 44 h if evaluation needs 8 h |
| Full validation selection, final evaluation, generation and scoring | Measured reserve, minimum 4 h; nominal 8 h |
| Reporting, checksums, archive, safe shutdown | Final 1 h |

Nominal total: 72 h; AE plus diffusion training up to 58 h. Aim for at least 48 productive training hours when throughput and convergence allow. Training/validation/checkpoint wall time is logged separately in metrics and stage records; do not label all allocated wall time as productive optimization.

The benchmark runs **inside the same allocation**. Measure scorer cost on generated outputs as well as real references. Multiply measured sampling/scoring/encoding/validation cost by 2.5; include all admitted runs, full valid/test counts, AE test reconstruction, scorer load allowance and reference groups. Reserve at least 4 hours plus the separate final hour. Admit four, otherwise two, otherwise one run only if each can receive at least two training hours; cap individual runs at 12 h. Persist the measured plan and per-run budgets before diffusion training. Increase evaluation reserve before admitting any follow-up. Preserve downstream generation/scoring reserves during final validation. Slow hardware reduces training, not the prescribed evaluation count. Each subprocess has a hard deadline; each training loop stops earlier for checkpoints. Once selection starts, recovery cannot admit new training runs.

Early convergence advances to the next predefined seed if at least two hours plus its evaluation reserve remain. Once all useful seeds finish, evaluate and finish early; never pad runtime. Saved per-run time and the total training pool prevent recovery from silently granting a new scientific training budget. A hardware failure or severe timing underestimate can still produce a partial result; `completion.json` is authoritative.

## Transfer and setup (no cluster job)

PowerShell on this computer:

```powershell
Set-Location C:\Users\SCIENTIST\Documents\GitHub\research\swahili_72h\transfer
Get-FileHash .\swahili72-transfer.tar -Algorithm SHA256
Get-Content .\swahili72-transfer.tar.sha256
ssh USER@CLUSTER_HOST "mkdir -p /datasets/hbutt/swahili72/slurm"
scp .\swahili72-transfer.tar .\swahili72-transfer.tar.sha256 USER@CLUSTER_HOST:/datasets/hbutt/swahili72/
ssh USER@CLUSTER_HOST
```

Then on the login node, file management/environment assembly only:

```bash
cd /datasets/hbutt/swahili72
sha256sum -c swahili72-transfer.tar.sha256
test ! -e package
tar -xf swahili72-transfer.tar
bash package/setup.sh
df -h /datasets/hbutt
quota -s
```

`setup.sh` reuses `$HOME/containers/anaconda-2025-12-2.sif` and Python `$HOME/.conda/envs/ld4lg-blackwell/bin/python` to create a separate environment at `/datasets/hbutt/swahili72/env`. It requires Python 3.10 and inherited torch 2.7.1+cu128. All other pinned dependencies are installed from 30 bundled Linux CPython 3.10 wheels using `--no-index`; no internet is required. No package is changed in the Afrikaans environment. The SIF and CUDA-enabled PyTorch are existing cluster prerequisites, not included in the transfer. If they are absent or versions differ, setup fails explicitly; resolve the environment before submission.

Allow **150 GiB free persistent space**, preferably 200 GiB including transfer, smoke checkpoints, caches, final checkpoints and uncompressed results archive; check quota as well as filesystem free space. Four CPUs/64 GiB RAM and one visible GPU in an exclusive node allocation. The launcher pins **mscluster110**, previously successful with RTX PRO 6000 Blackwell (~96 GB), without claiming current availability. No GRES is invented. mscluster107 had Apptainer namespace failure; mscluster111 had broken GPU telemetry; mscluster108 was historically down. If 110 is unavailable, wait or verify a substitute; do not silently switch to those nodes.

## ONE submission command

Only execute this when ready to launch; it has **not** been executed during preparation:

```bash
sbatch /datasets/hbutt/swahili72/package/job.slurm
```

Everything from GPU checks through final archive runs in this allocation. No dependencies, job chain, or separate preparation/benchmark/scoring allocation.

## Monitoring and download

Replace `JOBID` with the returned job number:

```bash
squeue -j JOBID
scontrol show job JOBID
tail -f /datasets/hbutt/swahili72/slurm/swahili72-JOBID.log
cat /datasets/hbutt/swahili72/results/stages.json
tail -n 30 /datasets/hbutt/swahili72/results/logs/ae_train.log
cat /datasets/hbutt/swahili72/results/plan.json
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
cat /datasets/hbutt/swahili72/results/completion.json
```

Per-stage logs persist in `results/logs/`; individual diffusion logs are `train-small-42.log`, etc. Metrics/checkpoints: `results/ae/` and `results/runs/MODEL-SEED/`. Cache: `results/cache/`. Final outputs: `results/results.json`, `RESULTS.md`, `results.png`, `results.pdf`, `blind-review.csv`, `review-key.csv`, and `results.tar` with checksums. Keep the review key away from reviewers. An archive may contain an honest partial outcome; check completion status.

PowerShell download:

```powershell
New-Item -ItemType Directory -Force C:\Users\SCIENTIST\Documents\GitHub\research\swahili_72h\downloaded | Out-Null
Set-Location C:\Users\SCIENTIST\Documents\GitHub\research\swahili_72h\downloaded
scp USER@CLUSTER_HOST:/datasets/hbutt/swahili72/results/results.tar .
scp USER@CLUSTER_HOST:/datasets/hbutt/swahili72/results/results.tar.sha256 .
Get-FileHash .\results.tar -Algorithm SHA256
Get-Content .\results.tar.sha256
tar -xf .\results.tar
```

## Contingency recovery only

After an interrupted allocation has ended, leave the package/config/assets/output paths unchanged:

```bash
sbatch /datasets/hbutt/swahili72/package/job.slurm --resume
```

This is recovery from interruption, not a planned second job. Stage receipts skip completed work; full latest checkpoints restore model, optimizer, EMA, RNG, data cursor and LR state. Atomic best/latest writes retain the last good checkpoint on failure. Latent caches resume from flushed cursor metadata; generation/scoring resume by saved sample index. The job lock prevents simultaneous writers. The checkpoint contract rejects changed code/data/config/assets. Signals request a clean stop; a supervisor enforces the scheduler end time and retains a final-hour archive reserve. An exhausted stage budget or failed quality gate needs diagnosis; rerunning identical failed work is not a remedy.

## Verification scope

See `VERIFICATION.md` and bundled `verification/`. Local RTX3060 and Windows tests are not claims about Blackwell throughput or cluster availability. Exact Linux container bindings, Slurm signal delivery, driver compatibility, filesystem bandwidth and measured stage feasibility are rechecked in the one allocation. No scientific claim is made from smoke-test outputs. The held-out test model evaluation has not been run locally.
