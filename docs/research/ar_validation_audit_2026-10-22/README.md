# Validation-only audit of the historical AR checkpoint

Run `ar-baseline-57759`, selected checkpoint step 60,000. Prepared 22 October 2026.

This investigation follows the [fixed protocol](PROTOCOL.md). It verifies the historical checkpoint and examines a bounded inference diagnostic before any decision on retraining. It does not implement a matched multi-seed AR experiment.

**Completed. Start with the [suitability report and reuse-versus-retrain decision](AR_SUITABILITY_REPORT.md).** The checkpoint is technically sound and useful as a historical contextual comparator. Its full-validation CE is 3.5949; primary-arm external mGPT PPL is 82.18. Those are different quantities. The original matched-AR hypothesis remains incompletely assessed.

Writing deliverables: completed-benchmark Results draft (`docs/research/ar_validation_audit_2026-10-22/COMPLETED_BENCHMARK_RESULTS_DRAFT.md` in the research repository, not included here) and [AR methods/results insertion](AR_METHODS_RESULTS_INSERT.md). The [AI-assisted inspection](AI_ASSISTED_INSPECTION.md) is separately labelled.

The [inference driver](run_audit.py) runs four separate stages: `preflight`, `validation`, `generate`, and `score`, using the existing repository `.venv/Scripts/python.exe`. New artifacts are written under outputs/ar-validation-audit-20261022 (`outputs/ar-validation-audit-20261022` in the research repository, not included here). The driver checks immutable input identities between stages and makes no training calls or test-file reads. Completed JSON summaries are protected against replacement; interrupted generation/scoring can resume their own new JSONL outputs.

The [analysis script](analyze_audit.py) generates comparison summaries, conditional bootstrap intervals and sample-review files. It reads the historical diffusion **validation** scores and the real-validation reference only. The blind-review sheet is unscored; no human study is implied.

The first preflight serialization issue and subsequent fixes are recorded in [execution notes](EXECUTION_NOTES.md). The strict load, checkpoint tensor equality and tokenizer parity checks passed. The stored preflight reproduces historical validation CE within approximately 0.00003. All prescribed inference stages and the conditional bootstrap analysis have finished.

This audit preserves the original 20% criterion as unassessed. Causal AR perplexity on real validation text, external mGPT perplexity on generated text, and latent velocity MSE measure different quantities.
