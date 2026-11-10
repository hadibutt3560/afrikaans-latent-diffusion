# Decoder adaptation of the last one or two blocks (job 73970)

Job package for the first decoder-adaptation experiment in §5.4 of the paper (Slurm job 73970, prepared 24 October 2026). The latent representation stayed frozen.

## Design

- **Start:** autoencoder job 56531, step 30500 (SHA-256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`).
- **Arms:** B0, the unchanged autoencoder; C1, last decoder block and final layer norm trainable; T2, last two blocks and final layer norm trainable. Encoder, Perceiver, embeddings, output projection, other decoder blocks, diffusion and tokenizer were frozen.
- **Training:** 47,767 NCHLT training passages with cached latents; teacher-forced cross-entropy; fresh AdamW per arm, learning rate 2e-5, weight decay 0.01, clipping 1.0, 100 warm-up updates; microbatch 1 × 8 accumulation; shared permutation, seed 20261024; FP32, dropout off; at most 3,000 updates per arm.
- **Decoding:** beam 4, length 5–128, no repeated trigrams, repetition penalty 1.2.
- **Stopping:** checks every 500 updates on the 81 development passages; after 1,000 updates, both arms stopped if neither gained 0.5 chrF over two consecutive checks.

## Thresholds fixed in advance

Eligible checkpoints had no empty or capped outputs, numeric precision and recall within 2 percentage points of B0, and repeated-trigram fraction within 0.005 of B0. Advancing to the 103-passage reserve required at least +3 chrF over B0 (67.7005) and at least 14/81 exact reconstructions; T2 also had to beat C1 by 1 chrF.

## Outcome

As recorded in the job 74177 plan, both arms stopped after 1,500 updates (elapsed 32:54) with no candidate. Best development chrF was 68.7525 (C1) and 68.8133 (T2) against 67.7005, with 12/81 exact against 11/81, so reserve and generation checks did not run. Full outputs are kept outside the repository.

## Files

- `job/train_decoder.py`: freeze mask, preflight, training, evaluation and selection.
- `job/decoder_refine.slurm`, `job/submit.sh`: Slurm script and submission wrapper.
- `job/test_logic.py`: synthetic CPU tests.

The plan fixed before the run is [PLAN.md](PLAN.md), unchanged; `train_decoder.py` checks its hash.
