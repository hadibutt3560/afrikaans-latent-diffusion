# Swahili latent diffusion: one 72-hour allocation

All files are in the separate `swahili_72h` workspace; Afrikaans files are unchanged.

## Experiment

- **Data:** Jordan's exact 664/37/37 file split, no text removed (see `DATA_AUDIT.md`: no document boundaries, some cross-split overlap). Lossless 48-word windows with stable IDs: 437,326 / 29,761 / 25,537 passages, none truncated at 192 mT5 tokens. Passage-level alignment needs the discrete pipeline to use the same exports.
- **Autoencoder:** pretrained mT5-base with a new 32-by-64 normalised Perceiver bottleneck; bottleneck and decoder trained on Swahili reconstruction, encoder, token embeddings and output projection frozen; the Afrikaans AE is not loaded. Weights match upstream LFS checksums (`assets-provenance.json`; [mT5](https://huggingface.co/google/mt5-base), [mGPT](https://huggingface.co/ai-forever/mGPT)). Selection by fixed validation token NLL. Gates before caching: token accuracy >=35%, mean character chrF >=25, NLL <=80% of the initialisation baseline. A failure yields diagnostic reconstructions and a partial archive; gates are not lowered.
- **Diffusion:** small (256 width, 4 layers) and published-size (768 width, 12 layers); cosine noise, velocity prediction, L2 loss, EMA .9999, original self-conditioning, DDPM250, beam-4 decoding. Model code is frozen in `vendor` with extraction provenance.
- **Runs:** small-42, published-42, small-43, published-43, then the same pair for seeds 44, 45, 46, as budget allows. A fixed 1,024-example MSE probe (three noise repeats) picks each run's best EMA checkpoint; full validation picks the winner. Every admitted run gets full test denoising evaluation and 256 validation-seeded plus 256 test-seeded unconditional generations (RNG banks, not conditioning). Test text is first encoded after `selection.json` freezes checkpoints.
- **Scoring:** pinned mGPT, FP32, context 512/stride 256, each token once with BOS, on all samples plus 256 real references per split; empty outputs stay visible. Reported: token-weighted and mean sample perplexity, diversity, lengths, losses. Perplexity is exploratory; mGPT pretraining overlap is unknown.
- **Human review** (pending): grammar 1 = mostly malformed, 5 = well-formed; coherence 1 = unrelated fragments, 5 = a consistent passage; natural Swahili 1 = largely unnatural, 5 = idiomatic. At least two fluent reviewers, randomised order, independent scores kept, labels hidden until scoring ends.

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

AE plus diffusion training may use up to 58 h, aiming for at least 48 productive hours. The in-allocation benchmark measures sampling, scoring (generated and real text), encoding and validation cost, multiplied by 2.5 for all admitted runs, full split counts, AE test reconstruction and references; at least 4 h plus the final hour are reserved. Four, else two, else one run is admitted only with at least two training hours each, capped at 12 h per run; the plan is saved before training. Reserves grow before follow-ups are admitted, and slow hardware cuts training, not evaluation. Subprocesses have hard deadlines. After early convergence the next seed starts only if two hours plus its reserve remain; the job ends early rather than padding. Recovery cannot add training budget or new runs after selection. `completion.json` records whether the result is complete.

## Transfer and setup

Local PowerShell:

```powershell
Set-Location C:\Users\SCIENTIST\Documents\GitHub\research\swahili_72h\transfer
Get-FileHash .\swahili72-transfer.tar -Algorithm SHA256
Get-Content .\swahili72-transfer.tar.sha256
ssh USER@CLUSTER_HOST "mkdir -p /datasets/hbutt/swahili72/slurm"
scp .\swahili72-transfer.tar .\swahili72-transfer.tar.sha256 USER@CLUSTER_HOST:/datasets/hbutt/swahili72/
ssh USER@CLUSTER_HOST
```

Login node:

```bash
cd /datasets/hbutt/swahili72
sha256sum -c swahili72-transfer.tar.sha256
test ! -e package
tar -xf swahili72-transfer.tar
bash package/setup.sh
df -h /datasets/hbutt
quota -s
```

`setup.sh` uses `$HOME/containers/anaconda-2025-12-2.sif` and `$HOME/.conda/envs/ld4lg-blackwell/bin/python` (Python 3.10, torch 2.7.1+cu128) to build `/datasets/hbutt/swahili72/env` offline from 30 bundled wheels (`--no-index`), and fails if the prerequisites differ. Needs 150 GiB free (preferably 200 GiB), four CPUs, 64 GiB RAM and one GPU on an exclusive node. The launcher pins **mscluster110**, previously successful with RTX PRO 6000 Blackwell (~96 GB); no GRES is set. Avoid mscluster107 (Apptainer namespace failure), mscluster111 (broken GPU telemetry) and mscluster108 (historically down).

## Submit, monitor and download

Single job; everything through the final archive runs inside it:

```bash
sbatch /datasets/hbutt/swahili72/package/job.slurm
```

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

Logs: `results/logs/` (e.g. `train-small-42.log`). Checkpoints: `results/ae/`, `results/runs/MODEL-SEED/`. Cache: `results/cache/`. Outputs: `results/results.json`, `RESULTS.md`, `results.png`, `results.pdf`, `blind-review.csv`, `review-key.csv` (withheld from reviewers) and `results.tar` with checksums.

```powershell
New-Item -ItemType Directory -Force C:\Users\SCIENTIST\Documents\GitHub\research\swahili_72h\downloaded | Out-Null
Set-Location C:\Users\SCIENTIST\Documents\GitHub\research\swahili_72h\downloaded
scp USER@CLUSTER_HOST:/datasets/hbutt/swahili72/results/results.tar .
scp USER@CLUSTER_HOST:/datasets/hbutt/swahili72/results/results.tar.sha256 .
Get-FileHash .\results.tar -Algorithm SHA256
Get-Content .\results.tar.sha256
tar -xf .\results.tar
```

## Recovery after interruption

With all paths unchanged, after the interrupted job has ended:

```bash
sbatch /datasets/hbutt/swahili72/package/job.slurm --resume
```

Stage receipts skip completed work. Checkpoints restore model, optimizer, EMA, RNG, data cursor and LR state, written atomically. Caches resume from flushed cursors and generation/scoring from the saved sample index. A lock prevents concurrent writers, and the checkpoint contract rejects changed code, data, config or assets. A watchdog process enforces the scheduler end time and keeps the final-hour archive reserve. A failed gate or exhausted budget needs diagnosis, not a rerun.

Local tests (`VERIFICATION.md`, `verification/`) do not establish Blackwell throughput, container bindings, signal delivery or stage feasibility; the allocation rechecks these.
