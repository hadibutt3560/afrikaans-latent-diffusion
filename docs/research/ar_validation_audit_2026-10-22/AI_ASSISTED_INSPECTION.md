# AI-assisted inspection of fixed AR validation samples

22 October 2026. **This is AI-assisted qualitative inspection, not human evaluation, a blinded comparison, or a validated Afrikaans fluency score.** No fluent human evaluator has supplied ratings. The observations below are diagnostic flags for competent review; they do not estimate the prevalence of linguistic errors in the generator's distribution.

The primary arm is `ar-cap127`, checkpoint `ar-baseline-57759/best_model`, selected step 60,000. Before scoring, 20 of its 256 outputs were selected by `random.Random(20261022).sample(range(256),20)`, as specified in the protocol. The reviewed generation seeds are 30180, 30144, 30088, 30024, 30123, 30117, 30120, 30001, 30067, 30030, 30101, 30099, 30157, 30054, 30076, 30059, 30110, 30230, 30002 and 30058. This selection was not based on mGPT scores.

Source: [primary-arm outputs](C:/Users/SCIENTIST/Documents/GitHub/research/outputs/ar-validation-audit-20261022/ar-cap127-samples.jsonl). The generated [inspection extract](C:/Users/SCIENTIST/Documents/GitHub/research/docs/research/ar_validation_audit_2026-10-22/SAMPLES_FOR_INSPECTION.md) contains the complete fixed sample plus separately labelled score-selected examples.

## Observations supported by the fixed sample

Some outputs have recognisable question or administrative-text forms. Seed30001 is “Wie kan 'n sertifikaat indien?” Seed30054 is “Die datum waarop die handelaar/wapensmid die aansoek doen om die lisensie moet in paragraaf F 5 aangeteken word.” These illustrate locally plausible forms; their presence does not establish general quality or originality. Exact copying is assessed separately by the mechanical training-match diagnostic.

Other outputs show fragmentary structure, unstable meaning or apparent malformed constructions:

| Seed | Observed feature | Interpretation limit |
|---|---|---|
| 30144 | “Dr Robert Ross U.” is a short name-like fragment. | Fragmentary does not automatically mean corpus-inappropriate; NCHLT contains passages rather than exclusively complete sentences. |
| 30099 | “Die datum waarop die verantwoordelike persoon moet in paragraaf D 17 aangeteken word.” appears to leave the embedded clause incomplete. | Flag for Afrikaans grammatical review; no numerical human rating is assigned. |
| 30030 | Lists “voertuie, voertuie” and ends as a fragment. | A low repeated-trigram score can miss repeated single words and structural incompleteness. |
| 30101 | Combines committee reporting, liquidation, constitutional/provincial references and copyright with unusual forms such as “Komiteee” and “verwysingsnoleed”. | Administrative vocabulary alone is insufficient evidence of coherent discourse. Spelling and syntax should be checked by a fluent evaluator. |
| 30059 | Combines vaccination, an electoral roll, SABS Design, antiretroviral medicines and fishing-related wording within one passage. | Indicates apparent topic blending and weak semantic coherence; no external factuality assessment was performed. |
| 30076 | Moves between children, agriculture, crime, a workshop and a small enterprise through unstable clause relationships. | Flags coherence rather than a failure to match an unrelated real reference. |

All 256 primary-arm outputs terminate with EOS, none is empty, none reaches the 127-new-token cap, and none contains a sentinel diagnostic flag. These mechanical successes coexist with the inspected linguistic problems. EOS termination is therefore not a certificate of sentence completeness or coherence.

## Consequence for the research claim

The checkpoint can produce recognisable Afrikaans-like administrative text and is not mechanically collapsed to empty or unterminated strings in this bank. It also produces outputs requiring substantial linguistic qualification. Its technical validity supports reuse as a historical comparator; it does not establish an adequate linguistic-quality standard against which diffusion can automatically be declared successful.

The independently saved review CSV has blank rating fields. Its existence is preparation for possible review, not evidence that human evaluation occurred. Any new human study must follow the agreed rubric, sampling plan, evaluator competence requirements and verified institutional requirements. The twenty-item AI inspection must remain separately labelled in the dissertation.
