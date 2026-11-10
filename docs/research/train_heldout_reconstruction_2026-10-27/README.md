# Train-versus-held-out reconstruction diagnostic — protocol

## Summary

- Tested: whether the original autoencoder's poor held-out reconstruction is mainly a failure to generalise, by comparing a training panel (T) with a held-out validation panel (V) of 194 passages each.
- Fixed in advance: G (generalisation-limited) needs exact_T ≥ 3 × exact_V, a lower 95% bound of Δexact ≥ +10 percentage points and a ΔCE interval above 0; F (fit-limited) needs exact_T ≤ 1.5 × exact_V and an upper bound ≤ +10 points; anything else is I.
- Outcome (`results/analysis.json`): I, indeterminate. Exact match 0.247 (T) against 0.134 (V); Δexact 0.113 [0.035, 0.191]; ratio 1.85.
- Results: `results/` (`analysis.json`, `ALL_RECONSTRUCTIONS.md`).

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), whose approval line was reworded on 10 November 2026; the exact version used by the run is at the git tag `as-run`, and `tools/verify_provenance.py` checks it against the hash in the run record.
