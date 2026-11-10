# Swahili checks: reconstruction, random latents, language and copying

## Summary

- Tested: the Afrikaans checks applied to the Swahili run (job 66417): reconstruction with a shuffled-latent control, a random-latent decoder control, language identification and copying.
- Fixed in advance: random latents are described as clearly worse only if their perplexity is above 1.20× the mean diffusion validation perplexity; all results are descriptive and on validation data.
- Outcome (`results/RESULTS.md`): reconstruction exact match 0/200 with chrF 74.8 (shuffled 24.3); random-latent perplexity 135.3 against a diffusion mean of 19.26 (ratio 7.03), so clearly worse; GlotLID gave `swh_Latn` for all outputs of the four diffusion banks (share 1.000).
- Results: `results/` (`RESULTS.md`, `summary.json`, `gates.json`, `s4.json`).

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.
