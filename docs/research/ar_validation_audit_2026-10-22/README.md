# Validation-only audit of the historical AR checkpoint

## Summary

- Tested: whether the historical AR checkpoint (`ar-baseline-57759`, step 60,000) is technically valid, and how a 127-token sampling cap compares with the historical 64-token cap.
- Fixed in advance: the reproduced selection-subset CE must lie within 0.02 of the recorded 3.6071377117359407 before any generation; no reselection; the proposal's 1.20 criterion is not marked as passed.
- Outcome (`AR_SUITABILITY_REPORT.md`): weights, tokeniser and validation CE verified (BF16 reproduction difference -0.000032438; full-validation CE 3.594901348). Primary-arm mGPT perplexity 82.1786 (64-token control 85.6705). Decision: reuse as a historical contextual comparator, not a matched AR experiment.
- Results: `AR_SUITABILITY_REPORT.md`, `VALIDATION_CHECKS.json` and `SAMPLES_FOR_INSPECTION.md`; raw outputs are in the research repository, not here.

The rules fixed before the run are in [PROTOCOL.md](PROTOCOL.md), kept exactly as it was when the study ran so that its SHA-256 still matches the run record.

Run `ar-baseline-57759`, selected checkpoint step 60,000. Prepared 22 October 2026.

This investigation followed the [fixed protocol](PROTOCOL.md). It verified the historical checkpoint and examined a bounded inference diagnostic. It did not implement a matched multi-seed AR experiment.

**Completed. Results are in the [suitability report](AR_SUITABILITY_REPORT.md).** The checkpoint is technically sound and was used as a historical contextual comparator. Its full-validation CE is 3.5949; primary-arm external mGPT PPL is 82.18. Those are different quantities. The original matched-AR hypothesis remains incompletely assessed. The report text was edited on 10 November 2026 for length and tone, with all measurements unchanged.

The [inference driver](run_audit.py) runs four separate stages: `preflight`, `validation`, `generate`, and `score`, using the existing repository `.venv/Scripts/python.exe`. New artifacts are written under `outputs/ar-validation-audit-20261022` in the research repository (not included here). The driver checks immutable input identities between stages and makes no training calls or test-file reads. Completed JSON summaries are protected against replacement; interrupted generation/scoring can resume their own new JSONL outputs.

The [analysis script](analyze_audit.py) generates comparison summaries, conditional bootstrap intervals and sample-review files. It reads the historical diffusion **validation** scores and the real-validation reference only. The blind-review sheet is unscored; no human study is implied.

The first preflight serialisation issue and subsequent fixes are recorded in [execution notes](EXECUTION_NOTES.md). The strict load, checkpoint tensor equality and tokenizer parity checks passed. The stored preflight reproduces historical validation CE within approximately 0.00003. All prescribed inference stages and the conditional bootstrap analysis finished.

This audit leaves the original 20% criterion unassessed. Causal AR perplexity on real validation text, external mGPT perplexity on generated text, and latent velocity MSE measure different quantities.
