# Follow-up inference checks: sampling audit, fitted prior and fresh reconstruction

## Summary

- Tested: F1, whether normalising the self-conditioning estimate at sampling changes outputs; F2, D's diffusion against a Gaussian fitted to D's training latents; F3, reconstruction on the 681 unseen `extra_monitor` passages.
- Fixed in advance: F1 shows no material effect only if every primary run's interval lies within ±0.10 nats per token; F2 counts as clearly worse above 1.20× the mean of the D published runs.
- Outcome (`analysis.json`): F1 material effect for D (pooled -0.212 nats per token) and inconclusive for B0; F2 fitted prior clearly worse (standardised perplexity 276.91 against 109.67); F3 exact match 415/681 for D, 171 for A and 13 for B0.
- Results: `analysis.json` in this folder. The protocol was not formally approved before the job ran (dated note of 4 November 2026).

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.
