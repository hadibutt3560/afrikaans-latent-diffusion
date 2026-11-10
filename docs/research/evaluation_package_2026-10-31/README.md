# Language validity, scorer sensitivity, AR decoding sensitivity and a checklist judge: protocol

## Summary

- Tested: language validity (GlotLID and an Afrikaans dictionary), shared phrasing with the training split, Goldfish as an alternative scorer, an AR sampling grid against the 1.20 criterion, and gated MAUVE and checklist-judge pilots.
- Fixed in advance: the tuned AR setting must clear the least diverse diffusion run; MAUVE gate G-M (human median at least 0.90, among five conditions); judge gate G-J (detection at least 0.80 per corruption type, clean flag rate at most 0.20, unparsed at most 0.02).
- Outcome: no AR grid cell was eligible, so no tuned-AR comparison was made (`results/status.json`); MAUVE failed its calibration gate (`results/mauve.json`); no judge candidate passed the selection half (`results/judge-selection.json`).
- Outcome (`results/analysis.json`): large minus small diffusion (C-1) was confirmed under both scorers; GlotLID gave `afr_Latn` as top label for a mean of 0.9632 (small) and 0.9724 (large) of diffusion outputs.
- Results: `results/` (`analysis.json`, `frontier.pdf`, `manual-check-result.json` and the gate files above).

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.
