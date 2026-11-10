# Local follow-ups to the 1 November review

## Summary

- Tested: descriptive follow-ups to the 1 November review: numeral errors, train–validation word 12-gram overlap, passage lengths, partial copying, scorer agreement, B0 on the 681 `extra_monitor` passages, shuffle-seed sensitivity and further random-latent banks.
- Fixed in advance: seeds and rules as below (for example shuffle seeds 20261111–20261115); no task tests a new hypothesis.
- Outcome (`results/SUMMARY.md`): every task finished. B0 reconstructed 13 of 681 `extra_monitor` passages exactly (chrF 65.64); 168 passages in 64 documents share a word 12-gram across training and non-reserve validation; shuffled latents gave 0 exact matches for all five seeds.
- Results: `results/` (`SUMMARY.md`, `status.json` and one JSON file per task).

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), whose approval line was reworded on 10 November 2026; the exact version used by the run is at the git tag `as-run`, and `tools/verify_provenance.py` checks it against the hash in the run record.
