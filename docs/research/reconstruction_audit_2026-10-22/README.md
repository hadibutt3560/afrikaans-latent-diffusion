# Step 1: frozen autoencoder reconstruction diagnostic

## Summary

- Tested: whether the frozen autoencoder (job 56531) uses passage-specific information, by comparing correct and cross-document shuffled latents on 81 validation passages.
- Fixed in advance: paired document bootstrap (10,000 replicates, seed 91428); correct-latent superiority with an interval excluding zero supports input dependence, not adequate reconstruction.
- Outcome (`RESULTS.md`): CE 1.0790 (correct) against 8.3189 (shuffled), difference 7.2399 [6.6049, 7.9331]; exact reconstruction 11/81 against 0/81; chrF 67.70 against 12.84. Information is retained, but adequate linguistic fidelity is not established.
- Results: `RESULTS.md` and `ALL_RECONSTRUCTIONS.md`.

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), whose approval line was reworded on 10 November 2026; the exact version used by the run is at the git tag `as-run`, and `tools/verify_provenance.py` checks it against the hash in the run record.
