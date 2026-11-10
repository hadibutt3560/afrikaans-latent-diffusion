# AR checkpoint suitability report

22 October 2026. The audit verified the historical AR checkpoint, ran a length-cap diagnostic and compared AR samples with the diffusion validation banks. No training was performed. The checkpoint was verified and used as a historical contextual comparator. It differs from the diffusion systems in pretraining, data exposure, compute, decoding, effective length and replication, so the matched-AR hypothesis was not fully tested; the criterion's reconstruction and degeneracy conditions were not assessed here.

## 1. Checkpoint verification

Checkpoint `outputs/cluster-results/ar-baseline-57759/best_model`, selected at step **60,000** (final step 157,529). The exported safetensors hash is `ed12edbd82757c6713c05eae442f95407f9ecf01fefaba97821b7a17c4b1d076`; its tensors exactly equal `best.pt` (SHA256 `edfab6128279f010046904ab7199d4bc3b4b358aabe99e998e872ae52e580000`). Strict loading reported no missing, unexpected or mismatched keys.

The randomly initialised model has 67,218,176 parameters, four layers, width 256, four heads and tied input/output embeddings. Its vocabulary hash matches the historical protocol, and the AR and diffusion tokenizers give identical truncated encodings for all 5,905 validation passages. BOS/PAD=0, EOS=1, the extra BOS position and attention and loss masking were checked.

| Validation measurement | Token-weighted cross-entropy |
|---|---:|
| Recorded historical selection subset, 128 passages | 3.607137712 |
| Same subset rerun, BF16 autocast | 3.607105273 |
| Same subset rerun, FP32 | 3.606869708 |
| Full validation, 5,905 passages, FP32 | 3.594901348 |

The BF16 reproduction difference is −0.000032438. The subset was rebuilt with Python seed 42, as in the training code, whose root and handoff copies have identical hashes. Of 630 logged checks on this subset, step 60,000 has the minimum CE; the last, at step 157,500, is 3.747406. Full validation rescored the selected checkpoint without reselecting it.

Full validation covers 296,627 target tokens from 251 documents. AR causal perplexity is **36.4121**; a 2,000-replicate document-cluster bootstrap gives CE **[3.2612, 3.9712]**, or perplexity **[26.0805, 53.0502]**, excluding training-seed uncertainty. This perplexity is not comparable with latent velocity MSE or external perplexity of generated text.

## 2. Length-cap diagnostic

Both arms use seeds 30000–30255, FP32, sampling from BOS with top-p 0.95, top-k 0, temperature 1, one beam and no repetition constraints. Only the cap differs: 127 new tokens (128 positions with BOS) against the historical 64.

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

All 256 pairs share token prefixes and 230 have identical text. Of the 28 capped control outputs, two end with EOS at the boundary and 26 do not. The paired interval for PPL(cap127) − PPL(cap64) is **[−5.3554, −1.5754]** around **−3.4920** (2,000 percentile replicates paired by seed): a score change, not evidence of better language. Cap127, declared in advance, is the primary comparator; no other caps were tested.

The 100 historical saved AR strings score **72.9632**, with mean length 18.12 words and two exact training matches; their seeds, precision and count differ.

## 3. Comparison with the diffusion validation banks

Each diffusion run contributes its first 256 validation-bank outputs (seeds 30000–30255). Equal seed numbers do not pair AR and diffusion outputs.

| System | Training seeds | Outputs per seed | External mGPT PPL | Mean words | Distinct-2 | Exact train matches |
|---|---:|---:|---:|---:|---:|---:|
| Historical AR, fresh cap127 bank | 1 | 256 | 82.18 | 17.53 | 0.81446 | 11/256 (4.30%) |
| Small diffusion | 5 | 256 | 53.07 ± 1.05 | 27.85 | 0.78311 | 1/1,280 (0.08%) |
| Large diffusion (`published`) | 5 | 256 | 50.02 ± 0.44 | 24.81 | 0.80603 | 26/1,280 (2.03%) |
| Real-validation reference | — | 256 total | 31.43 | 28.54 | 0.78908 | 0/256 |

Diffusion PPL is mean ± sample SD across training seeds; words and Distinct-2 are run means; copies are summed. Full 1,000-output PPLs are in `analysis.json`. These are validation banks, separate from the test-bank results.

The diffusion/AR ratios are **0.64584** (small) and **0.60872** (large), below 1.20, but the comparator is not matched as proposed, so the original criterion was not recorded as passed. The real-text row describes the scorer, not the AR denominator.

### Length bands

| Output words | AR cap127 PPL (n) | Small diffusion mean PPL | Large diffusion mean PPL |
|---|---:|---:|---:|
| 1–10 | 281.32 (90) | 346.00 | 205.52 |
| 11–20 | 78.99 (84) | 86.49 | 68.00 |
| 21–35 | 70.42 (54) | 54.43 | 47.56 |
| 36+ | 52.33 (28) | 40.48 | 39.13 |

Diffusion entries are means of five within-bank pooled PPLs. The small model has higher perplexity than AR in both shorter bands despite its aggregate advantage. The bands are descriptive, not a causal length adjustment.

## 4. Linguistic and copying observations

An AI-assisted reading of a fixed random 20 found recognisable Afrikaans administrative and question forms alongside incomplete constructions, repeated words and topic blending; seed 30059 mixes vaccination, an electoral roll, SABS Design, antiretroviral medicines and fishing. High-perplexity outputs include short fragments such as “Dis reg Edelagbare.” The 11 exact matches are short administrative or list-like strings such as “Rente op desentralisasie-voordele.”, counted per output, and stay in the scores. No human ratings were collected. Texts are in [SAMPLES_FOR_INSPECTION.md](SAMPLES_FOR_INSPECTION.md); a 60-item blind review sheet has blank ratings and a separately stored key.

## 5. Comparison conditions

| Comparison dimension | Existing AR | Completed diffusion | Remaining limitation |
|---|---|---|---|
| Pretraining | Random generator; inherited mT5 tokenizer | Pretrained mT5 representation/decoder, adapted on NCHLT | Unequal external data/knowledge; new AR training alone does not remove it. |
| Capacity | 67.218M total parameters, including large tied embeddings | 8.098M or 187.978M denoiser **plus** shared AE/decoder | Width equality is not parameter matching. |
| Local exposure | Replacement sampling, batch 8; selected 60k updates ≈480,000 rows; completed 157,529 ≈1,260,232 rows | Batch up to 64, shared 381-epoch maximum, configuration-dependent stopping | Realised examples and tokens at selected checkpoints differ. |
| Compute | Historical allowance 16,266s against earlier diffusion job 57231; actual timed phase 16,277s; selected-step log ≈6,292s | Final capacity suite has different training protocol and FP32 | Neither matched final-suite FLOPs nor full pipeline cost. |
| Selection | 128-example CE subset, 630 periodic checks, one training seed | Fixed 2,048-example velocity-MSE probe, repeated noise, five seeds | Different selection information and objectives. |
| Decoding | Top-p 0.95 sampling; no repetition ban; primary cap127 new tokens | DDPM 250 plus beam 4, minimum 5, repeated-trigram ban and penalty 1.2, total cap 128 | Different distributions and effective lengths remain even when maximum positions match. |

## 6. Reproducibility and resources

The [protocol](PROTOCOL.md) was fixed before inference; the [driver](run_audit.py), [analysis script](analyze_audit.py) and logs record the procedure. A JSON serialisation error was fixed without changing settings ([execution notes](EXECUTION_NOTES.md)).

Scoring used mGPT revision `40897bd7c8b47a76802c411108ca6220438b8b40`, FP32, context 512, stride 256 and the benchmark scoring function; four stored validation scores were reproduced with equal token counts and NLLs. Legacy `attn.masked_bias` keys reported by the loader (`scoring.log`) belong to an older GPT-2 attention buffer, not the AR checkpoint. The old script's moving default revision and mean-of-passage-PPL ratio were not used.

RTX 3060 12GB, PyTorch 2.11.0+cu128, Transformers 5.5.0. Core durations: **384.5s** validation, **140.9s** generation, **22.1s** scoring, about **9.1 minutes** combined, excluding loading, preflight/retry, anchors and analysis. Peak allocated CUDA memory: about **0.30 GiB** (generation) and **5.43 GiB** (scoring). No cluster job, training or model download was used, and neither script read test files.

Sources (research repository, not included here): `outputs/ar-validation-audit-20261022/` (`preflight.json`, `full-validation.json`, `analysis.json`, `comparison.csv`, `contract.json`, `scorer-anchors.json`, sample and scored files); `outputs/cluster-results/ar-baseline-57759/` (`best_selection.json`, `training_metrics.jsonl`); `outputs/longrun-local-evaluation/runs/{small,published}-seed{42,43,44,45,46}/validation-mgpt.json` and `validation-reference-mgpt.json`.
