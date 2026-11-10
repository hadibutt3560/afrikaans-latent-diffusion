# Diffusion latents against simpler latent priors — protocol

## Summary

- Tested: diffusion latents against three simpler latent priors (C1 isotropic, C2 fitted Gaussian, C3 recombined training slots), all decoded by the fixed B0 decoder, on validation.
- Fixed in advance: the primary contrasts are diffusion minus C3 in length-standardised mGPT perplexity, with approximate 97.5% t intervals; a bank with fewer than 20 outputs in any length band gets no standardised comparison.
- Outcome (`results/analysis.json`): both primary contrasts were insufficient support, because C3 had 2 outputs in the 1–10-word band. Exploratory contrasts: small and large diffusion lower than C1; small higher and large lower than C2 (`NOTE_2026-11-01_C3_support_and_C2.md`).
- Results: `results/` (`analysis.json`, `ALL_OUTPUTS.md`); reporting note `NOTE_2026-11-01_C3_support_and_C2.md`.

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.
