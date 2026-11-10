# Decoder dropout experiment (job 75003)

Job package for the third decoder-adaptation experiment in §5.4 of the paper (Slurm job 75003, prepared 25 October 2026). In job 74177, F12 development cross-entropy rose from 0.978 at step 4,920 to 1.337 at step 26,520 while training loss fell; this experiment tested whether decoder dropout reduced that overfitting.

## Design

- **Start:** autoencoder job 56531, step 30500 (SHA-256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`), fresh AdamW.
- **Arms:** D0, all 12 decoder blocks plus final layer norm, dropout off; D1, identical with decoder dropout 0.1 during training; B0, unchanged. Evaluation used dropout off. Encoder, Perceiver, embeddings, output projection and diffusion were frozen.
- **Training:** as job 74177 (AdamW 2e-5, weight decay 0.01, clipping 1, microbatch 1 × 8, FP32); both arms started from RNG seed 20261024. Decoding unchanged.
- **Budget and stopping:** one RTX PRO 6000 Blackwell on mscluster110, 16-hour Slurm limit, at most 13 hours of training; at most 26,620 updates per arm, checked at 1,320, 2,520, …, 26,620. After 12,000 updates, both arms stopped if neither gained 0.5 chrF over six successive checks.

## Thresholds fixed in advance

Eligibility as for jobs 73970 and 74177. Only D1 could advance: chrF at least 70.7005325922 (B0 + 3), at least 14/81 exact, and at least 1 chrF above D0. Reserve confirmation required +3 chrF with a 95% interval above zero and exact matches up by at least 0.03; D1 against D0 required +1 chrF with a positive interval.

## Outcome

Results are kept outside the repository. The paper (§5.4) reports that no decoder change passed its threshold: with dropout, development chrF stayed between 67.93 and 69.80, and the same-seed run without dropout tracked the earlier curve within 0.51.

## Files

- `job/train_dropout.py`: preflight, paired training, selection and conditional evaluation.
- `job/engine.py`, `job/training_support.py`, `job/transfer_eval.py`: helpers copied from job 74177 with edits; sources and hashes in `job/helper-provenance.json`.
- `job/export_report.py`, `job/dropout.slurm`, `job/submit.sh`, `job/test_logic.py`: export, Slurm script, wrapper, tests.
- `prepare_sources.py`: copies and edits the helpers.
- `verify_export.py`, `verify_low_memory.py`: local export and scorer checks.

The plan fixed before the run is [PLAN.md](PLAN.md), unchanged; `job/engine.py` checks its hash.
