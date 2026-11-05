# AR checkpoint suitability report

22 October 2026. **Decision: reuse this checkpoint as a historical contextual comparator. Do not retrain it merely to repair or replace an allegedly broken baseline: its weights, tokenizer and validation performance have now been verified.** A prospectively matched, replicated AR comparison still requires a separate agreed design and additional training unless suitable existing independent runs are recovered. No training was performed in this audit.

The current evidence favours diffusion on external mGPT perplexity in a bounded validation comparison, but does not complete the proposal's matched-AR hypothesis test. Pretraining, data exposure, compute, decoder constraints, effective length and replication differ. The original composite criterion also requires adequate reconstruction and no severe degeneracy; this audit does not establish those conditions.

## 1. What was verified

Checkpoint: `outputs/cluster-results/ar-baseline-57759/best_model`, selected step **60,000**, rather than the final training step 157,529. The exported safetensors hash is `ed12edbd82757c6713c05eae442f95407f9ecf01fefaba97821b7a17c4b1d076`. Its loaded tensors exactly equal `best.pt` (SHA256 `edfab6128279f010046904ab7199d4bc3b4b358aabe99e998e872ae52e580000`). Strict AR loading reports no missing, unexpected or mismatched keys.

The verified model has 67,218,176 parameters, four layers, width 256, four attention heads and tied input/output embeddings. It was randomly initialised; its inherited tokenizer does not imply pretrained generator weights. The vocabulary hash matches the historical protocol, and the AR and frozen diffusion tokenizers produce identical truncated encodings for all 5,905 validation passages. BOS/PAD=0, EOS=1, the extra BOS position, attention masking and loss masking were checked.

| Validation measurement | Token-weighted cross-entropy |
|---|---:|
| Recorded historical selection subset, 128 passages | 3.607137712 |
| Same subset rerun, BF16 autocast | 3.607105273 |
| Same subset rerun, FP32 | 3.606869708 |
| Full validation, 5,905 passages, FP32 | 3.594901348 |

The BF16 reproduction difference is −0.000032438. The subset is reconstructed by shuffling validation indices with Python seed 42, exactly as the preserved training code specifies. Root and handoff copies of that training script have identical hashes. The historical log contains 630 checks on this subset; step 60,000 has the minimum recorded validation CE. The last periodic check at step 157,500 is worse (3.747406). The full-validation pass rescored the already-selected checkpoint; it did not reselect it.

Full validation scores 296,627 target tokens from 251 source documents. AR's own causal perplexity is **36.4121**. A 2,000-replicate document-cluster percentile bootstrap gives CE interval **[3.2612, 3.9712]**, or perplexity **[26.0805, 53.0502]**. This interval reflects the empirical document mix conditional on the selected model; it is not AR training-seed uncertainty. Neither this causal perplexity nor reconstruction CE can be directly compared with latent velocity MSE or generated-text external perplexity.

Primary evidence: preflight (`outputs/ar-validation-audit-20261022/preflight.json` in the research repository, not included here), full-validation per-passage scores (`outputs/ar-validation-audit-20261022/full-validation.json` in the research repository, not included here), analysis and bootstrap results (`outputs/ar-validation-audit-20261022/analysis.json` in the research repository, not included here), historical selection (`outputs/cluster-results/ar-baseline-57759/best_selection.json` in the research repository, not included here), and training log (`outputs/cluster-results/ar-baseline-57759/training_metrics.jsonl` in the research repository, not included here).

## 2. The controlled length-cap diagnostic

Both new AR arms use 256 seeds (30000–30255), FP32, unconditional sampling from BOS, top-p 0.95, top-k 0, temperature 1, one beam and no repetition constraints. The primary arm allows 127 new tokens, or 128 total positions including BOS; the control uses the historical 64-new-token cap. Only the cap differs. Matching total positions to diffusion's decoder cap does not match decoding algorithms or realised text lengths. The distinction between total length and newly generated length follows the [Transformers generation API](https://huggingface.co/docs/transformers/v5.5.0/en/main_classes/text_generation).

| Diagnostic | Primary cap127 | Control cap64 |
|---|---:|---:|
| Samples | 256 | 256 |
| External pooled mGPT PPL | **82.1786** | **85.6705** |
| Conditional sample-bootstrap 95% interval | [74.7340, 90.9093] | [78.3089, 94.6829] |
| Mean whitespace words | 17.5273 | 16.6680 |
| EOS termination | 256/256 | 230/256 |
| Reaches length cap | 0/256 | 28/256 |
| Empty outputs / sentinel flags | 0 / 0 | 0 / 0 |
| Exact normalised training-text matches | 11/256 | 11/256 |
| Distinct-2 | 0.81446 | 0.81501 |

All 256 pairs share identical token prefixes; 230 pairs have identical complete decoded text. Of the 28 control outputs reaching the cap, two terminate with EOS at that boundary and 26 do not. Increasing the cap therefore removes artificial nontermination for 26 outputs in this bank. The paired conditional bootstrap interval for PPL(cap127) − PPL(cap64) is **[−5.3554, −1.5754]**, around a point difference of **−3.4920**. These are 2,000 fixed-seed percentile bootstrap replicates, paired by AR generation seed.

**Interpretation:** the old cap is a real inference limitation for some outputs, and removing it changes the external score modestly. This is an inference diagnostic with a measured score improvement, not evidence of better linguistic quality. All primary-arm outputs finish naturally before the larger cap, yet qualitative problems remain. Stop this cap investigation here; do not search additional caps for a favourable score. Use the predeclared cap127 arm as this audit's primary contextual comparator, retaining cap64 as sensitivity evidence.

The 100 historical saved AR strings score **72.9632** under the same scorer, with mean length18.12 words and two exact training matches. Their generation seeds, precision/runtime and sample count differ from the new arms. They are provenance/context, not another controlled cap treatment or an independently trained replicate.

Sources: new primary samples (`outputs/ar-validation-audit-20261022/ar-cap127-samples.jsonl` in the research repository, not included here), new control samples (`outputs/ar-validation-audit-20261022/ar-cap64-samples.jsonl` in the research repository, not included here), corresponding `*-scored.jsonl` files, and the paired analysis (`outputs/ar-validation-audit-20261022/analysis.json` in the research repository, not included here).

## 3. Contextual comparison with the completed diffusion validation banks

For each of the five diffusion training seeds per configuration, take the first256 stored validation-bank outputs (generation seeds30000–30255). This matches the number of outputs per run; it does not match training or length distributions. The same numerical seed on AR and diffusion is not a semantic pairing and is not used as one in a cross-model statistical test.

| System | Training seeds | Outputs per seed | External mGPT PPL | Mean words | Distinct-2 | Exact train matches |
|---|---:|---:|---:|---:|---:|---:|
| Historical AR, fresh cap127 bank | 1 | 256 | 82.18 | 17.53 | 0.81446 | 11/256 (4.30%) |
| Small diffusion | 5 | 256 | 53.07 ± 1.05 | 27.85 | 0.78311 | 1/1,280 (0.08%) |
| Large diffusion (`published`) | 5 | 256 | 50.02 ± 0.44 | 24.81 | 0.80603 | 26/1,280 (2.03%) |
| Real-validation reference | — | 256 total | 31.43 | 28.54 | 0.78908 | 0/256 |

Diffusion PPL is mean ± sample SD across training seeds. Its word count and distinct-2 are run means; copying counts sum outputs. AR's conditional sample interval is in the preceding table and must not be mistaken for comparable training-seed SD. Full 1,000-output diffusion-bank PPLs are preserved in `analysis.json` as context. This table uses only validation generation banks and must not be combined with the previously published-in-project test-bank results as though they came from one evaluation.

The descriptive diffusion/AR ratios are **0.64584** (small) and **0.60872** (large). They lie below 1.20 in this contextual validation comparison. **Do not report the original criterion as passed.** The comparator is not matched as proposed, its own linguistic quality is limited, and the reconstruction/degeneracy requirements have not been established. The real-text row is context for the scorer; it is neither the trained AR denominator nor a correct reference for an individual unconditional output.

### Length sensitivity changes the interpretation

The fixed word-length bands show that the aggregate ordering does not hold uniformly for the small diffusion model:

| Output words | AR cap127 PPL (n) | Small diffusion mean PPL | Large diffusion mean PPL |
|---|---:|---:|---:|
| 1–10 | 281.32 (90) | 346.00 | 205.52 |
| 11–20 | 78.99 (84) | 86.49 | 68.00 |
| 21–35 | 70.42 (54) | 54.43 | 47.56 |
| 36+ | 52.33 (28) | 40.48 | 39.13 |

Diffusion entries are means of five within-bank pooled PPLs, not pooled values across all runs; per-seed band counts are retained in `analysis.json`. The small model's aggregate advantage coexists with higher average perplexity in both shorter bands. These are coarse descriptive strata, not a causal length adjustment: topic, text type and remaining within-band lengths still differ. They argue against reducing the comparison to one ratio. Do not length-filter outputs out of the primary result after observing these scores.

Sources: comparison CSV (`outputs/ar-validation-audit-20261022/comparison.csv` in the research repository, not included here), analysis (`outputs/ar-validation-audit-20261022/analysis.json` in the research repository, not included here), and historical artifacts `outputs/longrun-local-evaluation/runs/{small,published}-seed{42,43,44,45,46}/validation-mgpt.json`, plus `validation-reference-mgpt.json` in that directory.

## 4. Linguistic and copying diagnostics

The fixed random20 inspection contains recognisable Afrikaans administrative/question forms alongside incomplete constructions, repeated words and topic blending. For example, seed 30059 combines vaccination, an electoral roll, SABS Design, antiretroviral medicines and fishing-related wording. All primary-arm outputs have EOS; this does not establish coherent clauses. The high-perplexity examples include short fragments and “Dis reg Edelagbare.”, illustrating why external PPL is not a grammaticality label.

Exact matches include short administrative or list-like strings such as “Rente op desentralisasie-voordele.” and “Dui u antwoord met 'n X in paragraaf F 13 aan.” The count is 11 generated outputs, including repeated outputs; it is not a count of unique memorised facts. All remain in the primary scores. No private-information memorisation claim is made.

These observations are **AI-assisted**, not human ratings. See [labelled inspection](AI_ASSISTED_INSPECTION.md) and [fixed sample with targeted examples](SAMPLES_FOR_INSPECTION.md). The 60-item blind review sheet (`outputs/ar-validation-audit-20261022/unrated-blind-review.csv` in the research repository, not included here) has blank ratings and is only a preparation artifact; its key (`outputs/ar-validation-audit-20261022/review-key.csv` in the research repository, not included here) must be withheld from prospective evaluators. It does not replace the previously proposed full human-evaluation protocol or institutional determination.

## 5. Reuse versus retrain

**Reuse now:** write up this checkpoint as a technically verified historical contextual baseline, with all limitations in the comparison table. No further AR training is necessary to support that limited claim. The current investigation is complete; do not expand it into a decoding sweep.

**If the original comparative RQ is retained:** agree the matching target with the supervisor before new training. A practical option is a prospectively declared downstream-compute comparison with inherited pretraining and AE adaptation explicitly accounted for or excluded by a named scope convention. Match frozen corpus/view, validation-selection allowance, evaluation sample counts and the primary scoring protocol; document total/trainable/generation-active parameter counts and realised token exposure. Use at least three new AR training seeds, preferably five if allocation permits, and report uncertainty accordingly. Historical seed 42 is not a replicate of a changed training protocol. This is a proposed minimum replication plan, not a claim that three seeds automatically give sufficient statistical power.

**If new matched training is not feasible:** seek supervisor agreement to retain the representation/capacity diagnostic study with this contextual comparator, and explicitly state that the original matched hypothesis remains incompletely tested. Do not silently replace it with the easier numerical comparison above.

| Comparison dimension | Existing AR | Completed diffusion | Remaining limitation |
|---|---|---|---|
| Pretraining | Random generator; inherited mT5 tokenizer | Pretrained mT5 representation/decoder, adapted on NCHLT | Unequal external data/knowledge; new AR training alone does not remove it. |
| Capacity | 67.218M total parameters, including large tied embeddings | 8.098M or187.978M denoiser **plus** shared AE/decoder | Width equality is not parameter matching; full system counts still need a consolidated table. |
| Local exposure | Replacement sampling, batch 8; selected60k updates ≈480,000 rows; completed157,529 ≈1,260,232 rows | Batch up to 64, shared381-epoch maximum, configuration-dependent stopping | Count realised examples and non-padding tokens at selected checkpoints, not only final training steps. |
| Compute | Historical allowance 16,266s against earlier diffusion job57231; actual timed phase 16,277s; selected-step log≈6,292s | Final capacity suite has different training protocol and FP32 | Neither matched final-suite FLOPs nor full pipeline cost. |
| Selection | 128-example CE subset, 630 periodic checks, one training seed | Fixed2,048-example velocity-MSE probe, repeated noise, five seeds | Different selection information and objectives; full-valid rescoring does not undo selection history. |
| Decoding | Top-p0.95 sampling; no repetition ban; primary cap127 new tokens | DDPM 250 plus beam 4, minimum 5, repeated-trigram ban and penalty 1.2, total cap128 | Different distributions and effective lengths remain even when maximum positions match. |

**Recommended next substantive experiment:** the previously planned correct-versus-shuffled latent reconstruction audit on the frozen AE. It targets the strongest unresolved bottleneck and helps answer RQ2. Decide the AR matching/scope issue in parallel; do not spend compute on a larger diffusion model based on this contextual score gap.

## 6. Reproducibility, limitations and resources

The [fixed protocol](PROTOCOL.md) was written before fresh inference. The [driver](run_audit.py), [analysis script](analyze_audit.py), input/runtime contract (`outputs/ar-validation-audit-20261022/contract.json` in the research repository, not included here), per-text/per-document outputs and execution logs preserve the procedure. An initial JSON serialization error was repaired without changing the protocol or numerical settings; it is disclosed in [execution notes](EXECUTION_NOTES.md).

The pinned mGPT revision is `40897bd7c8b47a76802c411108ca6220438b8b40`. Four fixed stored validation scores were recomputed with exactly equal token counts and NLLs; see anchor checks (`outputs/ar-validation-audit-20261022/scorer-anchors.json` in the research repository, not included here). The mGPT loader reports unexpected legacy `attn.masked_bias` keys; the report is retained in `scoring.log`. These names correspond to a nonpersistent attention-mask buffer in the available older GPT2 implementation, rather than an AR checkpoint mismatch. The anchor agreement supports scorer compatibility on checked examples, not a universal bitwise guarantee. The preserved benchmark scoring function, FP32, context 512 and stride 256 were reused; the old comparison script's default moving revision and arithmetic mean-of-passage-PPL ratio were not used.

Hardware/software: existing RTX 3060 12GB, PyTorch 2.11.0+cu128, Transformers 5.5.0. Recorded successful core durations were **384.5s** for full validation, **140.9s** for generation and **22.1s** for AR scoring, approximately **9.1 minutes combined**, excluding checkpoint loading, preflight/retry, anchors and analysis. Peak allocated CUDA memory was about **0.30 GiB** during generation and **5.43 GiB** during scoring; these are allocator peaks, not total system/driver VRAM. No cluster job, model download, paid compute, package installation or training was used. Test files were not read by either audit script; historical test exposure remains a project-level limitation.

The completed-benchmark Results draft (`docs/research/ar_validation_audit_2026-10-22/COMPLETED_BENCHMARK_RESULTS_DRAFT.md` in the research repository, not included here) is ready for author revision and keeps the earlier benchmark distinct from this exploratory validation comparison.
