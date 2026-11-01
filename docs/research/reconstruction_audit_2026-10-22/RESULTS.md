# Step 1 completed: the frozen autoencoder uses input information but loses content

The correct latent representation strongly outperforms a shuffled representation. The autoencoder is therefore using passage-specific information. However, reconstruction remains imperfect and sometimes changes important content. This supports a qualified RQ2 answer: information is retained, but adequate linguistic fidelity has not been established. This diagnostic does not isolate the encoder, compression layer or decoder as the cause.

## Verified experiment

Final frozen AE: job56531, step30500, [checkpoint args](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/cluster-results/afrikaans-56531/best/args.json), model SHA256 `123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c`. Loaded strictly using archived source from `outputs/longrun-local-inputs`. The existing16-example [latent diagnostic](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/cluster-results/afrikaans-56531/latent-diagnostic.json) and eight-example [interface check](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/nchlt-afrikaans/interface-audit-20260930/summary.json) informed the need for a larger, documented reconstruction contrast; they are not pooled into this result.

The [prospective protocol](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/reconstruction_audit_2026-10-22/PROTOCOL.md) selected64 documents, yielding81 passages under the up-to-two-per-document cap. This is below128 because many documents contain one passage; no replacements were made. Length strata contain44 short,32 medium and5 long passages. The document-balanced design is not a passage-uniform estimate for the full validation corpus. A separate103-passage/64-document reserve was recorded but not evaluated: the diagnostic contrast was clear and did not require confirmation or tuning.

Correct and cross-document shuffled latents were evaluated in FP32 with the same deterministic beam4 decoder, maximum128 positions, minimum5, repetition penalty1.2 and repeated-trigram prohibition. All81 main inputs were untruncated and their detokenized model views exactly matched source strings. Two overlength validation passages were examined separately. Validation has been exposed previously; this is exploratory diagnostic evidence, not a new untouched benchmark.

The [manifest](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/manifest.json) records exact passage/document IDs, bins and donors. The [contract](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/contract.json) records source, protocol, script and checkpoint hashes, settings and package versions.

## Results

Primary values: [analysis.json](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/analysis.json), derived from all81 rows of [examples.jsonl](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/examples.jsonl).

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

Teacher forcing provides the true preceding tokens at every step; its79.91% accuracy is not free-generation accuracy. Numeric strings use a regex and multiset matching; clock times and spaced numbers may contribute multiple strings. Matching these strings does not establish correct numeric meaning: row27 preserves R250000 while changing per-year to per-month. chrF and character similarity measure overlap, not grammatical or semantic adequacy.

The shuffled-minus-correct CE difference is7.2399 nats/token; a paired recipient-document bootstrap with10000 replicates gives95% interval[6.6049,7.9331]. The correct-minus-shuffled accuracy difference is59.71 percentage points, interval[55.90,63.72]. Mean sentence chrF difference is54.86, interval[48.07,61.33]. These intervals support strong dependence on the correct input under this fixed donor mapping. They exclude AE-training variability and alternative donor mappings; shared donors create dependence beyond recipient-document clustering. They are conditional diagnostics, not population-wide or causal-component confidence guarantees.

Correct-latent chrF by source length is66.45 (44 short),71.87 (32 medium),51.998 (5 long). The small long-input count prevents a strong length-effect claim. Errors are not restricted to long inputs.

## Content preservation

[AI-assisted inspection](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/reconstruction_audit_2026-10-22/AI_ASSISTED_REVIEW.md) covers every correct reconstruction, with full text in [ALL_RECONSTRUCTIONS.md](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/reconstruction_audit_2026-10-22/ALL_RECONSTRUCTIONS.md). No fluent-human evaluation was conducted. Observed problems include:

- Row20: competitor name, dates and venue changed; unsupported World Bank wording added.
- Row27: annual payment changed to monthly payment and one negation marker omitted.
- Row50: named hospitals and hearing aids replaced by information-centre wording.
- Row1: a short service-delivery fragment replaced by website wording.
- Row58: document-requirement sentence copied exactly, showing that not all outcomes fail.

These are illustrative cases within a complete inventory, not estimates of semantic-error prevalence. Some source passages are English, other-language text, navigation or metadata. Some discrepancies are spelling/punctuation corrections rather than meaning loss. Nevertheless, clear content failures also occur in recognisable Afrikaans prose. Therefore neither noisy data nor exact-match strictness alone explains the observed limitation.

The separate [truncation appendix](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/truncation-appendix.json) contains IDs `nchlt-afr-seg-9c6cce361298c7d517b829fc38dd70784c36f57ffa74885e14c889e92931954f` (144 tokens) and `nchlt-afr-seg-ab12867a96a30600b4f57dfd0c325fdaaf1ec06647a21ed3605c5803563024b4` (173 tokens). Their correct-output chrF against model view/source is41.34/38.64 and42.33/34.61 respectively. Truncation removes source content in these cases, while poor model-view reconstruction indicates additional loss. They are excluded from the81-passage headline table.

## Integrity, runtime and stopping decision

[Checks](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/checks.json) pass: unique/count-consistent IDs; unchanged protocol/driver/checkpoint/data; historical tokenizer/cache identity; finite scores; valid same-bin, cross-document derangement; disjoint reserved documents. Every shuffled generation exactly matches its donor's correct reconstruction, an additional check that the intended latent was supplied. The base-model loader reports newly initialized adapter parameters before the trained AE state is strictly loaded; no missing trained parameters remain after that strict load.

[Runtime](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/reconstruction-audit-20261022/runtime.json):224.70 seconds (about3.75 minutes) for encoding and diagnostics after loading, peak allocated GPU memory2.55GiB, NVIDIA RTX3060. Loading and CPU analysis are excluded from that duration. The local Python launcher required execution outside the filesystem sandbox; the same existing environment was used. No training, paid compute, cluster job or test access occurred. Historical outputs were not modified.

Step1 is complete. The result answers the diagnostic question sufficiently; no repeat, new decoder setting or confirmation run is warranted for this step. Adequate reconstruction is not a passed gate, and the cause is not uniquely identified. The rest of the research plan is outside this task.

## Ready-to-use Results paragraph

To assess whether the frozen autoencoder retained passage-specific information, we compared correct and cross-document shuffled latent representations on81 validation passages from64 documents. The same deterministic decoder was used in both conditions. Correct conditioning reduced token-weighted teacher-forced cross-entropy from8.319 to1.079 and increased mean sentence chrF from12.84 to67.70. A paired document bootstrap placed the shuffled-minus-correct cross-entropy difference at7.240 nats/token (95% interval6.605–7.933), conditional on the selected documents and fixed donor assignment. However, only11 of81 passages were reconstructed exactly, and21 of33 source numeric strings were retained. AI-assisted source–output inspection identified changes to entities, temporal expressions, negation and clause content; this inspection does not constitute fluent-human evaluation. None of the main diagnostic inputs was truncated. The results support passage-specific information retention but reveal a reconstruction-fidelity limitation under the evaluated decoder, without identifying which autoencoder component is responsible.
