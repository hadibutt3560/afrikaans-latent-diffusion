# Stage 3: bounded decoder improvement investigation

## Summary

- Tested: whether removing the repeated-trigram ban (`no_repeat_ngram_size` 3 to 0) improves reconstruction of the 81 Stage 1 passages.
- Fixed in advance: advance only if mean chrF improves by at least 2 points with a paired 95% interval entirely above 0, and numeric recall falls by no more than 2 percentage points.
- Outcome (`RESULTS.md`): chrF +0.9185 [-0.1104, 2.0160] and numeric recall down 6.06 percentage points. The gate failed, the reserve was not used and the original decoder settings were kept.
- Results: `RESULTS.md`, `CHANGED_PAIRS.md` and `PROJECTION_ADMISSION.md`.

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), whose approval line was reworded on 10 November 2026; the exact version used by the run is at the git tag `as-run`, and `tools/verify_provenance.py` checks it against the hash in the run record.
