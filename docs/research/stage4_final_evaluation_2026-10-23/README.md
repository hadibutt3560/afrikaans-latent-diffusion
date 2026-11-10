# Stage 4: proposal-aligned final evaluation

## Summary

- Tested: proposal-aligned automatic evaluation of the frozen models from existing outputs: test-bank perplexity, diversity and copying, the validation AR ratio, and external perplexity of reconstructions.
- Fixed in advance: 10000 bootstrap replicates with RNG 20261023; the large-minus-small contrast and the AR ratio are exploratory, not significance tests; no new training or generation.
- Outcome (`FINAL_EVALUATION.md`): large-minus-small test perplexity -2.7040 (95% interval -4.0757 to -1.3342); diffusion/AR validation ratios 0.64584 (small) and 0.60872 (large); composite hypothesis not established.
- Results: `FINAL_EVALUATION.md`, `analysis.json`, `test_seed_table.csv` and `validation_comparison.csv`.

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.
