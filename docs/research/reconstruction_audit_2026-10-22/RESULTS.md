# Reconstruction audit: correct against shuffled latents

The frozen autoencoder (job 56531, step 30500, SHA-256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`) was evaluated on 81 validation passages from 64 documents (44 short, 32 medium, 5 long; at most two per document, no replacements), selected under the [protocol](PROTOCOL.md). Correct and cross-document shuffled latents were decoded in FP32 with the same deterministic decoder (beam 4, length 5–128, repetition penalty 1.2, no repeated trigrams). All 81 inputs were untruncated. A separate 103-passage, 64-document reserve was recorded but not evaluated. Validation had been exposed before, so this is an exploratory diagnostic, not a new benchmark. Outputs (`manifest.json`, `contract.json`, `analysis.json`, `examples.jsonl`, `checks.json`, `runtime.json`, `truncation-appendix.json`) are in `outputs/reconstruction-audit-20261022/` in the research repository, not here; every integrity check passed, and each shuffled output exactly matched its donor's correct reconstruction.

The correct latent strongly outperforms the shuffled one, so passage-specific information is used, but reconstruction is imperfect and sometimes changes content. The diagnostic does not identify whether the encoder, compression layer or decoder is responsible. Runtime was 224.70 s on an RTX 3060 (peak 2.55 GiB); no training or test access took place.

| Measure | Correct latent | Shuffled latent |
|---|---:|---:|
| Teacher-forced token-weighted cross-entropy, lower better | 1.0790 | 8.3189 |
| Teacher-forced token accuracy | 79.91% | 20.20% |
| Exact free reconstruction | 11/81 (13.58%) | 0/81 |
| Mean sentence chrF, 0–100 | 67.70 | 12.84 |
| Mean character SequenceMatcher | 0.7731 | 0.2127 |
| Matched source numeric strings | 21/33 (63.64%) | 0/33 |
| Matched / generated numeric strings | 21/25 (84.00%) | 0/25 |
| EOS reached | 81/81 | 81/81 |
| Output cap reached | 0/81 | 0/81 |

Paired recipient-document bootstrap (10,000 replicates), conditional on the fixed donor mapping:

| Difference | Estimate | 95% interval |
|---|---:|---|
| Cross-entropy, shuffled − correct (nats/token) | 7.2399 | [6.6049, 7.9331] |
| Token accuracy, correct − shuffled (points) | 59.71 | [55.90, 63.72] |
| Mean sentence chrF, correct − shuffled | 54.86 | [48.07, 61.33] |

| Source length | n | Correct-latent chrF |
|---|---:|---:|
| Short | 44 | 66.45 |
| Medium | 32 | 71.87 |
| Long | 5 | 51.998 |

Notes:

- Teacher-forced accuracy is not free-generation accuracy. Numeric matching is regex multiset matching and does not establish meaning (row 27 keeps R250000 but changes per year to per month).
- Every reconstruction is in [ALL_RECONSTRUCTIONS.md](ALL_RECONSTRUCTIONS.md); the content checks reported in the paper are in [claude_labels_2026-11-07](../claude_labels_2026-11-07/). Examples: row 20 (name, dates and venue changed), row 27 (annual to monthly, negation dropped), row 50 (hospitals and hearing aids replaced), row 1 (fragment replaced by website wording), row 58 (copied exactly). No fluent-human evaluation was done.
- Two overlength passages were examined separately and excluded from the table: `nchlt-afr-seg-9c6cce361298c7d517b829fc38dd70784c36f57ffa74885e14c889e92931954f` (144 tokens; chrF 41.34 against model view, 38.64 against source) and `nchlt-afr-seg-ab12867a96a30600b4f57dfd0c325fdaaf1ec06647a21ed3605c5803563024b4` (173 tokens; 42.33 / 34.61).
