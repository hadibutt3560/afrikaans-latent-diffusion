# AR baseline on the test split: common-split check of the 1.20 rule

## Summary

- Tested: the AR baseline on the test split (1,000 samples), so that it and the original (B0) and retrained (D) diffusion systems are compared on one split under the 1.20 rule.
- Fixed in advance: a system passes if its perplexity ratio to AR is at most 1.20; it clearly passes if the upper interval limit is also at most 1.20, clearly fails if the lower limit is above 1.20, and is otherwise inconclusive.
- Outcome (`results/analysis.json`): AR test perplexity 79.15 (standardised 79.47). B0 clearly passes on raw and standardised perplexity (raw ratios 0.648 small, 0.614 large). D small clearly fails (raw ratio 3.644); D large is inconclusive on raw perplexity (1.212) and clearly fails on standardised perplexity (1.380).
- Results: `results/` (`analysis.json`, `anchors.json`, `contract.json`, `length_bands_ar.json`).

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.
