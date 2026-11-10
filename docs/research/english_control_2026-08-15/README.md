# English control: BART autoencoder on ROCStories (run 15 August 2026)

This folder holds the source for the RQ1 numbers in the paper: 93.26% validation token accuracy, and 11 of 128 exact reconstructions (8.59%).

- **Run:** `autoencoder-batch8-effective32-5k`.
  - `facebook/bart-base`, with the backbone frozen and latents of 32 × 64;
  - micro-batch 8 with gradient accumulation 4 (effective batch 32), 5,000 updates, BF16, on an RTX 3060;
  - ROCStories with 93,161 training stories, evaluated on 128 validation and 128 test stories.
- **Step 5,000:**
  - validation: cross-entropy 0.239, token accuracy 93.26%, exact reconstruction 11 of 128;
  - test: cross-entropy 0.265, token accuracy 93.11%, exact reconstruction 10 of 128.
- **Validation curve:** every 500 updates. Exact reconstruction was still rising at the end of the run (4.7% at update 4,000, 10.2% at 4,500, 8.6% at 5,000), so the reduced run had not converged.

`english_control_metrics.json` was extracted on 5 November 2026 from the project's research audit of 20 September 2026 (`evaluation_series.json` and `experiment_register.csv` in the private research repository). It contains every ROCStories evaluation and register row. The original per-step JSON files and weights are not in this repository.
