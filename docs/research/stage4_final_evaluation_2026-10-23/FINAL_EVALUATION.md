# Stage 4: final automatic evaluation

23 October 2026. **Automatic evaluation is consolidated; no new model or decoding setting is adopted.** Human evaluation is excluded by the submitted proposal (PDF p24, printed p20). Stage 4 involved no training or fresh generation. Test data had been inspected before, so this is dated secondary analysis of exposed artefacts. The failed Stage 3 intervention appears in no table. Artefacts: [analysis](analysis.json), [source hashes](source_hashes.json), [protocol](PROTOCOL.md), [checks](checks.json). All ten stored test PPLs were recomputed from individual NLL/token totals.

## Diffusion test benchmark

Each of five training seeds per configuration (one frozen AE and corpus split) contributes 1000 unconditional outputs. Values are run means ± sample SD; per-seed results and sample-bootstrap intervals are in [test_seed_table.csv](test_seed_table.csv).

| Measure | Small | Large |
|---|---:|---:|
| External mGPT PPL | 51.3115 ±1.4850 | 48.6075 ±0.9094 |
| Test velocity MSE | 0.501131 ±0.000290 | 0.451224 ±0.000179 |
| Distinct1 | 0.22294 ±0.00390 | 0.23431 ±0.00223 |
| Distinct2 | 0.66059 ±0.00513 | 0.69027 ±0.00283 |
| Mean words | 29.296 ±1.259 | 26.343 ±0.517 |
| Mean within-output repeated trigram fraction | 0.00001296 | 0.00002914 |
| Mean within-output repeated fourgram fraction | 0 | 0.00000377 |
| Exact normalized training matches | 5/5000 | 79/5000 |
| Sum of within-run normalized duplicate excess | 2 | 1 |
| Empty outputs | 0/5000 | 0/5000 |
| EOS reached | 4995/5000 | 4994/5000 |
| Length cap reached | 6/5000 | 6/5000 |
| Sentinel flags | 2/5000 | 0/5000 |

Large-minus-small mean PPL is −2.7040 (−5.27%), unpaired training-run bootstrap 95% interval [−4.0757, −1.3342]: exploratory, with five runs per group and differing realised budgets. Text-bootstrap intervals in the CSV exclude training variability. Bootstraps use 10000 replicates and RNG 20261023, with no output pairing or multiple testing.

Words are case-sensitive whitespace tokens. Repetition is the per-output fraction of n-gram occurrences beyond unique n-grams, averaged over outputs. Copying and duplicates use NFC, case folding and collapsed whitespace, summed within runs. Token-trigram suppression already limits repetition, EOS and cap counts can overlap, and length differences confound diversity.

## AR comparison: validation only

The 256-output AR bank (one training seed) is compared with the first 256 stored validation outputs of every diffusion run.

| System | External PPL | Diffusion/AR ratio | Conditional hierarchical95% ratio interval |
|---|---:|---:|---|
| AR | 82.1786 | — | — |
| Small, five-run mean | 53.0745 | 0.64584 | 0.58130–0.71479 |
| Large, five-run mean | 50.0238 | 0.60872 | 0.54774–0.67358 |

Intervals resample diffusion runs and samples plus AR samples, not AR- or AE-training variability. Geometric-mean ratios are 0.64574 and 0.60870; one-sided 95% upper bounds 0.70463 and 0.66287 (supplementary, not the proposal's decision rule). Per-run detail: [validation_comparison.csv](validation_comparison.csv). The Stage 2 budget reconciliation in [ar_scope_decision_2026-10-22](../ar_scope_decision_2026-10-22/) gives small diffusion loop costs of 2.93–3.13 h against 4.52 h for AR on the same recorded GPU model, with different precision and extra AE costs. Compute, pretraining and selection are therefore unmatched, and the comparison is reported as qualified.

## External scorer stability

Stored real-text pooled PPL is 33.6224 for the 1000-example test reference and 32.8193 for the 1000-example validation reference. By whitespace-word length:

| Words | Real test PPL (n) | Real validation PPL (n) |
|---|---:|---:|
| 1–10 | 240.83 (157) | 161.05 (140) |
| 11–20 | 54.09 (212) | 56.18 (223) |
| 21–35 | 32.60 (300) | 32.96 (349) |
| 36+ | 25.93 (331) | 24.46 (288) |

mGPT is length- and domain-sensitive, and small diffusion's advantage over AR reverses in the two shorter bands of the validation comparison. It remains the proposal's primary automatic metric but is read as a **weak linguistic-quality diagnostic**. Per-run [length bands](length_bands.json) are supplied; no outputs were discarded.

## Reconstruction

Historical AE evaluation covers 1024 passages per split:

| Measure | Validation | Test |
|---|---:|---:|
| Reconstruction CE | 0.775916 | 0.857749 |
| Teacher-forced token accuracy | 82.75% | 80.59% |
| Exact reconstruction | 12.21% | 7.23% |
| Numeric recall | 57.60% | 50.95% |

Stage 1's separate set of 81 passages from 64 documents gave correct-latent CE 1.0790 against 8.3189 shuffled, 11 exact reconstructions, and entity, number, negation and clause changes. Stage 4 scored those 81 reconstructions and their sources with mGPT: pooled PPL **70.6728 against 61.7604**, paired-document bootstrap 95% interval for the log-PPL difference [0.00115, 0.25842]. This is a scorer-preference contrast, not a fidelity test, and the document-balanced sample is not comparable with the 1000-reference sample.

Scores and the contract with four reproduced anchors: `outputs/stage4-final-evaluation-20261023/reconstruction-scores.jsonl` and `scoring-contract.json` (research repository only). Settings: mGPT revision `40897bd7c8b47a76802c411108ca6220438b8b40`, FP32, context 512/stride 256, archived scorer. A first attempt stopped on an invalid encoding alias, corrected to `utf-8-sig`; both logs are kept. Legacy mask-buffer warnings match the documented scorer limitation, and all four anchors passed.

## Tokenisation and inspection

The tokenizer report (`data/nchlt_afrikaans/analysis/mt5-base-tokenizer/report.json`, research repository only) gives validation mean fertility 1.9331 including special tokens, median 1.875, p95 = 2.5, zero unknown tokens and 2/5905 inputs above 128 tokens. On Stage 1's 81 passages (fertility without EOS), Spearman correlations are 0.122 with reconstruction CE, 0.254 with reconstruction PPL and −0.032 with source token length; these are not causal or population estimates. The proposed English fertility distribution was not recovered; English/ROCStories substitution remains an RQ1 deviation.

The proposal's checks (entities and content, recognisable clauses, unsupported additions) were applied to the 81 reconstructions as an **AI-assisted** reading by Claude: [claude_labels_2026-11-07](../claude_labels_2026-11-07/). No human agreement is reported.

## Research questions

| Item | Final evidence status |
|---|---|
| RQ1 English implementation control | Reduced ROCStories/BART controls documented; OpenWebText evidence and the stronger reported run's artefacts incompletely verified. Qualified, not full fulfilment. |
| RQ2 sufficient representation for reconstruction | Source-specific information retained, but content-preservation failures mean adequacy is not established. |
| RQ3 credible AR comparison | External-PPL ratios below 1.20 on validation; budget, pretraining and selection differences prevent a matched interpretation. |
| No severe degeneracy | No widespread empty-output or literal n-gram collapse under constrained decoding; copying and content/coherence failures remain. |
| Composite hypothesis | **Not established.** Favourable aggregate PPL cannot override matching and reconstruction limitations; this is not a proven intrinsic failure of diffusion. |

Remaining deviations are documented limitations, not missing experiments.
