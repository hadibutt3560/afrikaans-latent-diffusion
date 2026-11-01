# Stage 4: final automatic evaluation

23 October 2026. **Automatic evaluation is consolidated; no new model or decoding setting is adopted.** Human evaluation is not conducted, as explicitly excluded by the submitted proposal (PDF p24, printed p20) and reaffirmed by the user. Researcher-side inspection must not be confused with participant ratings or AI-assisted inspection.

## Evidence and interpretation

This report separates the completed diffusion test benchmark from the later validation AR comparison and reconstruction diagnostics. Test data/output inspection occurred historically. This is dated secondary analysis of exposed artifacts, not a new untouched test. The failed Stage3 intervention is not substituted into any final table. No training or fresh generation occurred in Stage4.

The [analysis](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/analysis.json), [source hashes](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/source_hashes.json), [protocol](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/PROTOCOL.md) and [checks](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/checks.json) identify the underlying artifacts. All ten stored test PPLs were recomputed from individual NLL/token totals. All five training seeds are retained per configuration, conditional on one frozen AE and corpus split.

## Completed diffusion benchmark

Each training run contributes1000 unconditional outputs. Values below are arithmetic run means ± sample SD over five training seeds, not uncertainty among5000 independent trained models. See [all seed results and sample-bootstrap intervals](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/test_seed_table.csv).

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

Large-minus-small mean PPL is−2.7040 (−5.27%); the unpaired training-run bootstrap95% interval is[−4.0757,−1.3342]. This exploratory interval uses only five runs/group and is not a confirmatory causal capacity effect: realised budgets differ. Per-model text-bootstrap intervals are a separate uncertainty layer in the CSV; they do not incorporate training variability. All bootstrap analyses use10000 replicates and fixed RNG20261023. No arbitrary output pairing or multiple significance tests are used.

Diversity/repetition here uses case-sensitive whitespace words, identically across systems. Repetition is the per-output fraction of n-gram occurrences beyond unique n-grams, averaged over outputs. Exact copying and duplicates use NFC, casefolding and collapsed whitespace; duplicate totals are summed within runs, not calculated across a pooled5000-output bank. Token-trigram suppression already constrains repetition, so low literal repetition does not establish coherent language. EOS and cap counts can overlap. Short administrative matches do not independently prove harmful memorisation. Length differences confound diversity interpretation.

## AR comparison: validation only

The existing256-output AR bank is compared with the first256 stored validation outputs from every diffusion run. This table is not the test table above. The verified AR checkpoint is a single training seed; its own likelihood is not the external mGPT score.

| System | External PPL | Diffusion/AR ratio | Conditional hierarchical95% ratio interval |
|---|---:|---:|---|
| AR | 82.1786 | — | — |
| Small, five-run mean | 53.0745 | 0.64584 | 0.58130–0.71479 |
| Large, five-run mean | 50.0238 | 0.60872 | 0.54774–0.67358 |

Intervals resample diffusion runs and their samples plus samples from the one fixed AR checkpoint. They cannot estimate AR-training or AE-training variability. Geometric-mean ratios are0.64574 and0.60870; one-sided95% upper bounds are0.70463 and0.66287. These supplementary interval summaries were not the proposal's original decision rule. Being below1.20 does not resolve unmatched compute/pretraining, unequal selection or reconstruction/degeneracy requirements.

[Validation per-run table](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/validation_comparison.csv) includes PPL uncertainty, lengths, Distinct1/2, repetition and copying. The [Stage2 budget reconciliation](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/ar_scope_decision_2026-10-22/STAGE_2_COMPLETED.md) remains authoritative: small diffusion loop costs2.93–3.13h versus AR4.52h on the same recorded GPU model, with different precision and additional AE costs. This is not equal end-to-end compute. Retain the original question and report a qualified comparison; no new scope amendment is presumed.

## External scorer stability and limitations

Stored real-text pooled PPL is33.6224 for the1000-example test reference and32.8193 for the1000-example validation reference. By whitespace-word length:

| Words | Real test PPL (n) | Real validation PPL (n) |
|---|---:|---:|
| 1–10 | 240.83 (157) | 161.05 (140) |
| 11–20 | 54.09 (212) | 56.18 (223) |
| 21–35 | 32.60 (300) | 32.96 (349) |
| 36+ | 25.93 (331) | 24.46 (288) |

These substantial differences show length/domain sensitivity; no externally justified numerical stability threshold was specified. Small diffusion's aggregate advantage over AR reverses in the two shorter word-length bands in the existing validation comparison. Therefore retain mGPT as the proposal's named primary automatic metric but interpret it as a **weak linguistic-quality diagnostic**, with the sensitivity visible. It is not a reliable standalone grammaticality/meaning judge. Full per-run [length bands](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/stage4_final_evaluation_2026-10-23/length_bands.json) are supplied. No primary outputs were discarded to improve scores. Real unrelated text is a distributional reference, never a per-output reconstruction target.

## Reconstruction, including the missing external score

Historical AE evaluation covers1024 passages per split:

| Measure | Validation | Test |
|---|---:|---:|
| Reconstruction CE | 0.775916 | 0.857749 |
| Teacher-forced token accuracy | 82.75% | 80.59% |
| Exact reconstruction | 12.21% | 7.23% |
| Numeric recall | 57.60% | 50.95% |

These historical subsets are distinct from Stage1's81 passages/64 documents. Stage1 showed correct-latent CE1.0790 versus shuffled8.3189,11 exact reconstructions, and documented entity/number/negation/clause changes. Lower teacher-forced loss does not ensure faithful free decoding; velocity MSE is not discrete-model likelihood.

To complete the proposal's requirement to score reconstructions externally, Stage4 scored those81 saved correct reconstructions and their matching source views. Pooled mGPT PPL is **70.6728 for reconstructions versus61.7604 for sources**. The paired-document bootstrap95% interval for reconstruction-minus-source log PPL is[0.00115,0.25842]. This is a scorer preference contrast, not a fidelity test or an independent confirmation population. The sample is document-balanced and includes short/noisy text, so its score is not directly comparable with the1000-reference corpus sample.

Primary new evidence: [81 scored pairs](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/stage4-final-evaluation-20261023/reconstruction-scores.jsonl), [scoring contract and four reproduced anchors](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/stage4-final-evaluation-20261023/scoring-contract.json). The pinned mGPT revision is `40897bd7c8b47a76802c411108ca6220438b8b40`, FP32, context512/stride256, using the archived scorer. The first attempt stopped before scoring because of an invalid encoding alias; it was corrected to `utf-8-sig` and rerun. Both logs are retained. No numerical protocol changed. The loader's legacy mask-buffer warnings match the earlier documented scorer compatibility limitation; four anchors passed before new scoring.

## Tokenisation and sample inspection

The [Afrikaans tokenizer report](C:/Users/SCIENTIST/Documents/GitHub/research/data/nchlt_afrikaans/analysis/mt5-base-tokenizer/report.json) reports validation mean fertility1.9331, median1.875, p95=2.5, zero unknown tokens and2/5905 inputs above128 tokens. This fertility includes special tokens. Stage1's exploratory fertility subtracts the single EOS token; under that different documented convention its Spearman correlations are0.122 with reconstruction CE,0.254 with external reconstruction PPL and−0.032 with source token length. These associations on81 selected passages are neither causal tokenizer effects nor population estimates. The proposed English fertility distribution was not recovered in the bounded evidence search; do not fabricate it or substitute Afrikaans statistics. English/ROCStories substitution remains an RQ1 deviation.

The proposal asks for researcher-side checks of major entities/content, recognisable clauses and unsupported lexical additions. The [Stage1 inspection](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/reconstruction_audit_2026-10-22/AI_ASSISTED_REVIEW.md) and [AR/generated-sample inspection](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/ar_validation_audit_2026-10-22/AI_ASSISTED_INSPECTION.md) provide traceable **AI-assisted** observations. They are not automatically the author's personal judgement. The author must review and own those interpretations before submission; no fabricated language competence or human agreement is reported.

## Proposal criterion and research-question conclusions

| Item | Final evidence status |
|---|---|
| RQ1 English implementation control | Reduced ROCStories/BART controls documented; proposed OpenWebText evidence and stronger reported run's primary artifacts remain incompletely verified. Report a qualified implementation-control account, not full fulfilment of the proposed setting. |
| RQ2 sufficient representation for reconstruction | Source-specific information is retained. Content-preservation failures mean adequacy is not established; later diffusion quality claims must remain limited. |
| RQ3 credible AR comparison | Numerical external-PPL ratios are below1.20 for the disclosed validation comparison. Budget/pretraining/selection differences prevent a fully matched interpretation. |
| No severe degeneracy | No widespread empty-output or literal n-gram collapse under constrained decoding; copying and content/coherence failures remain. No broad linguistic-quality pass is certified. |
| Composite hypothesis | **Not established.** Favourable aggregate PPL cannot override matching and reconstruction limitations. This is not a proven intrinsic failure of diffusion or a reason to hide mixed results. |

Stage4's automatic consolidation is complete. Remaining source/proposal deviations and author interpretation are explicitly documented limitations, not completed missing experiments. Human evaluation is excluded under the author's confirmed proposal scope; it is not a pending Stage4 deliverable. The next stage is manuscript completion, with these limits carried forward.
