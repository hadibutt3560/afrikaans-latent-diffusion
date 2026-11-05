# Tokenisation fertility from saved outputs (5 November 2026)

**Approval:** asked for in chat on 5 November 2026 as one of the fixes after two external reviews, which noted that RQ3 names tokenisation fertility but the paper reported only one mean. The definitions below were fixed in `run_fertility.py` before it was run. No model is trained or sampled; only saved texts are tokenised.

## Definitions
- **Fertility** is the number of mT5 tokens per whitespace word, without special tokens, pooled over a bank (total tokens divided by total words).
- **Tokeniser:** the mT5-base tokeniser used by every system (`outputs/longrun-local-inputs/models/mt5-base` in the research repository).

## Readouts
1. **Fertility of each test bank:**
   - human test text;
   - the 1,000 AR test samples;
   - the original diffusion, 5 runs per size;
   - the retrained diffusion, 3 runs per size (mean, minimum and maximum over runs).
2. **Reconstruction by source fertility,** on the 681 extra-data passages, for the original autoencoder and arms A and D:
   - exact-match rate and mean chrF in each third of passages, split by source fertility;
   - Spearman correlation between fertility and chrF;
   - Spearman correlation between fertility and source length, because fertility and length move together.

## Outputs
`results/fertility.json`. The script reads its inputs from the research repository layout; pass `--research` to point elsewhere.
