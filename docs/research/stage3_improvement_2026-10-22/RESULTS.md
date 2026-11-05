# Stage 3 completed: retain the original decoder

Finalised23 October2026; folder name retains the date the work started. **Removing the repeated-token-trigram restriction did not meet the predefined improvement criteria. Keep the original decoder settings.** One new controlled intervention was tested; terminal projection was reviewed and deferred. No training, final evaluation or dissertation-wide edits were performed.

## Experiment and integrity

The [protocol](PROTOCOL.md) was saved before treatment inference. The hypothesis was that the repeated-trigram prohibition obstructed faithful reconstruction. We used the final frozen AE job56531/step30500, the same81 validation passages from64 documents as Stage1, and the same encoded latent for each paired reconstruction. These are historically exposed validation data, not an untouched benchmark.

The only changed setting was `no_repeat_ngram_size: 3 -> 0`. Both arms used FP32, batch1, deterministic beam4 decoding, maximum128 positions, minimum5, repetition penalty1.2 and length penalty1. All81 baseline token sequences reproduced Stage1 exactly. This tests one decoding restriction, not whether the whole decoder or latent representation is optimal.

Exact IDs, full outputs and per-passage metrics: pairs.jsonl (`outputs/stage3-decoder-20261022/pairs.jsonl` in the research repository, not included here). Input/protocol/source identities and runtime settings: contract.json (`outputs/stage3-decoder-20261022/contract.json` in the research repository, not included here). Aggregate results: analysis.json (`outputs/stage3-decoder-20261022/analysis.json` in the research repository, not included here). Verification (`outputs/stage3-decoder-20261022/checks.json` in the research repository, not included here) confirms parent reproduction, unchanged frozen inputs, aggregate recomputation and complete inspection coverage.

## Results

| Metric | Original restriction | Restriction removed |
|---|---:|---:|
| Mean sentence chrF | 67.7005 | 68.6190 |
| Exact reconstruction | 11/81 | 14/81 |
| Mean character SequenceMatcher | 0.77313 | 0.77442 |
| Source numeric strings retained | 21/33 (63.64%) | 19/33 (57.58%) |
| Numeric-string precision | 21/25 (84.00%) | 19/23 (82.61%) |
| Excess repeated word trigrams, summed over outputs | 0 | 80 |
| Excess repeated word fourgrams, summed over outputs | 0 | 57 |
| EOS reached | 81/81 | 81/81 |
| Output cap reached | 0/81 | 0/81 |

chrF uses SacreBLEU2.6.0: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`. The mean improvement is **0.9185 points**, with paired document-bootstrap95% interval **[-0.1104,2.0160]** (10000 replicates, seed91428). The interval reflects selected validation documents conditional on one AE, not AE-training variability. It does not establish equivalence or prove zero effect.

The predefined gate required at least2 chrF points, an interval entirely above zero, and no more than a2-percentage-point decline in numeric recall. The treatment fails all three: the point gain is below2, the interval includes zero, and numeric recall drops **6.06 percentage points**. Therefore the reserve and unconditional-transfer stages are not admitted. This decision follows the frozen rule; no thresholds were changed after seeing outcomes.

Repeated n-grams here are case-sensitive whitespace-word tuples, counted within each output as occurrences beyond the first and summed over outputs. They are not a count of80 defective passages. The decoder restriction operates on model tokens, so these word-based counts are a separate diagnostic. Numeric strings use the Stage1 regex/multiset measure and do not guarantee preservation of numeric meaning.

## Inspection and interpretation

Thirty pairs changed in token sequence;29 changed in decoded text. All are documented in [AI-assisted inspection](AI_ASSISTED_INSPECTION.md) and [changed-pair text](CHANGED_PAIRS.md). The other51 are token-identical and retain the Stage1 inspection. No human ratings are claimed.

Useful corrections occur: row11 restores an apostrophe, row21 restores the second PHONE placeholder, and row56 restores the trauma/psychological-assessment sentence exactly. These explain the increase in exact copies. However, row41 repeats `I mfpamba:`, row48 repeats `Zola`, and row57 loses all clock-time content while repeating a weekday phrase. Other overlap gains retain wrong names, conditions or clause meanings. A progress message prematurely said repetition continued to the output limit; the completed measurements correct that: **all outputs terminated before the cap**.

The experiment shows that the restriction can obstruct some faithful copies, but simply removing it is not a defensible general fix under the predefined criteria. It does not isolate encoder versus compression versus decoder causes, and it does not show that all decoder interventions would fail. No claim about improved unconditional generation follows from this reconstruction test.

## Projection decision and stopping

[PROJECTION_ADMISSION.md](PROJECTION_ADMISSION.md) records the prior decoder-only projection experiment: PPL ratios1.0115 and1.0241 at50 and250 steps, with both intervals including1. No saved final-benchmark generated-latent bank or norm summaries were located. Final-model norm mismatch therefore remains unmeasured. We deferred fresh projection work as lower priority; we did not claim that a final-model norm gate was passed or that projection can never help.

**Stage3 decision:** retain `no_repeat_ngram_size=3` and the existing benchmark. Do not use the103-passage reserve, run unconditional transfer, start another decoder sweep, or retrain the AE on this evidence. This is a completed bounded investigation with a negative/mixed outcome, not an incomplete search for a positive result. The larger earlier optional programme is not represented as fully executed.

The diagnostic took **197.10 seconds after model loading**, with **2.38GiB peak allocated GPU memory** on the RTX3060. Loading/CPU documentation time is excluded. Protocol/script/source identities are recorded; the existing environment required the same execution permission used in Stage1. All results are in new paths and historical outputs are preserved.

## Manuscript-ready paragraph

We tested whether the decoder's repeated-token-trigram prohibition limited reconstruction fidelity by removing that constraint while holding all other settings fixed. On81 validation passages from64 documents, mean sentence chrF increased from67.70 to68.62, a difference of0.92 points (paired document-bootstrap95% interval−0.11 to2.02). Exact reconstructions increased from11 to14, but numeric-string recall decreased from63.64% to57.58%, and repeated word sequences increased. All outputs terminated before the maximum length. The intervention failed the prospectively specified improvement gate and was not advanced to confirmation or unconditional-generation evaluation. We retained the original decoder. This mixed result suggests that removing the constraint can recover some repeated source content but also exposes repetitive decoding failures; it does not identify a general solution to the reconstruction limitation.
