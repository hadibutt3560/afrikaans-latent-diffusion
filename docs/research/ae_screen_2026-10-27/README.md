# Autoencoder recipe, extra data and latent length: screen and diffusion protocol

## Summary

- Tested: four autoencoder arms using the full-mT5 recipe (with or without about 511k words of extra Afrikaans; 32 or 64 latent slots) against the job 56531 autoencoder (B0), with diffusion trained only on the winner.
- Fixed in advance: on the reserve panel the winner counts as improved only with at least +10 points exact match and +3 chrF (both with 95% lower bounds above 0) and numeral recall no more than 0.02 below B0.
- Outcome (`screen_report_from_cluster.json`): arm D won. On the reserve panel, exact match rose from 19/103 to 73/103 (difference 0.524, 95% interval 0.425 to 0.633), chrF rose by 30.38 (24.66 to 36.61) and numeral recall was 1.0 against 0.774 for B0.
- Generation comparison (section 7): see the paper.
- Results: `screen_report_from_cluster.json` in this folder; dated note `NOTE_2026-11-01_empty_bands.md`.

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), whose approval line was reworded on 10 November 2026; the exact version used by the run is at the git tag `as-run`, and `tools/verify_provenance.py` checks it against the hash in the run record.
