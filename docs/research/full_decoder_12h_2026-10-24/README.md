# Full-decoder adaptation (job 74177)

Job package for the second decoder-adaptation experiment in §5.4 of the paper (Slurm job 74177, prepared 24 October 2026, results retrieved 25 October 2026). Training, reserve evaluation, conditional generation, mGPT scoring and export ran as one job.

## Design

- **Start:** autoencoder job 56531, step 30500 (SHA-256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`), not job 73970 weights.
- **Arms:** F12, all 12 decoder blocks plus final layer norm; C2, last two blocks plus final layer norm; B0, unchanged. Encoder, Perceiver, embeddings, output projection and diffusion were frozen.
- **Training:** as job 73970 (AdamW 2e-5, weight decay 0.01, clipping 1, microbatch 1 × 8, FP32, dropout off), with permutations seeded 20261024 + epoch. Decoding unchanged.
- **Budget and stopping:** one RTX 8000 on mscluster107, 12-hour Slurm limit, 10.5-hour training phase, no plateau stopping; aborts for non-finite loss, changed frozen weights or more than 36 GiB allocated.

## Thresholds fixed in advance

Eligibility as for job 73970. Advancing to the reserve required at least 70.70 chrF (B0 67.70 + 3) and at least 14/81 exact, with F12 at least 1 chrF above C2. Reserve confirmation required +3 chrF with a positive 95% interval and +3 percentage points exact; only then would 256 shared latents from small-seed42 be decoded and scored.

## Outcome

From `RETRIEVED_JOB_74177.md`: Slurm COMPLETED (exit 0:0, elapsed 10:09:29); 26,620 updates per arm (about 4.46 corpus passes); 8.720 training hours. The selected F12 checkpoint, step 26,520, reached chrF 70.0539 against 67.7005, 13/81 exact against 11/81 and 25/33 numeric matches against 21/33. This was below the thresholds, so reserve and generation evaluation were skipped. Full outputs and large checkpoints are kept outside the repository.

## Files

- `job/train_full.py`: preflight, paired training, evaluation and selection.
- `job/transfer_eval.py`, `job/export_report.py`: generation scoring and export.
- `job/full_decoder.slurm`, `job/submit.sh`, `job/test_logic.py`: Slurm script, wrapper, synthetic tests.
- `verify_export.py`: local export check.

`engine.py`, imported by `train_full.py`, is not here; an edited copy is in `../decoder_dropout_final_2026-10-25/job/`. The plan fixed before the run is [PLAN.md](PLAN.md), unchanged.
