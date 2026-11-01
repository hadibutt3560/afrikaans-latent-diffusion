# Changed reconstruction pairs: AI-assisted inspection

This is source–baseline–treatment inspection by an AI assistant, not a blinded fluent-human assessment or a quantified language-quality score. Indices are the frozen Stage1 manifest positions; exact IDs and full outputs are in `outputs/stage3-decoder-20261022/pairs.jsonl`. Unchanged outputs retain the Stage1 inspection. Character-overlap gains do not certify preserved meaning. Source noise/mixed languages are retained.

| Index | Observed effect of removing the token-trigram restriction |
|---|---|
| 0 | Taxi-office wording changes, but false payment/commission content persists. |
| 4 | Some study wording returns; source male scholarships become female scholarships. Meaning remains wrong despite overlap gain. |
| 5 | UNESCO intrusion removed and Stellenbosch restored, but university names become merged/repeated. |
| 9 | Application wording returns, with an extra review clause; not a faithful reconstruction. |
| 11 | Exact source apostrophe restored. |
| 15 | Source first-person pronoun restored; source extra `van` still omitted. |
| 17 | English region phrase repeats; geographic fidelity remains poor. |
| 19 | Competition wording restored locally, but names and medal clauses repeat. |
| 20 | More repeated tournament/venue clauses with unsupported dates/locations; overlap worsens. |
| 21 | Second PHONE placeholder restored; exact reconstruction. |
| 26 | Malformed affordability wording improved; human/quality concepts still changed. |
| 29 | Token sequence differs but decoded text is identical; source person/meaning errors remain unchanged. |
| 30 | Some source question wording restored, but mixed-language navigation and invented token remain. No competent full syntax judgement. |
| 33 | Public Information repeats; question's subject/employment content still altered. |
| 34 | Non-Afrikaans input: visibly repeated words/phrases increase; no competent full semantic/syntax rating. |
| 40 | Environmental-management wording restored locally; original requirement clauses still altered. |
| 41 | A repeated `I mfpamba:` phrase dominates the treatment, losing the source question. Clear repetitive failure. |
| 45 | Tour-guide phrase repeats in the purpose clause; database still omitted. Overlap gain does not repair meaning. |
| 47 | Applicability wording repeats; curriculum-delivery phrase still altered. |
| 48 | `Zola` repeats extensively and question content is lost/altered. Clear repetitive failure. |
| 49 | Small-business wording restored instead of small farmers; laws still encourage rather than affect businesses. |
| 51 | Production-area phrase repeats; names and contact information still corrupted. |
| 55 | Non-governmental-organisation phrase repeats and abbreviations disappear; nonprofit concept remains corrupted. |
| 56 | Exact source restored, including trauma/psychological assessment. |
| 57 | Weekday phrase repeats and all clock-time content disappears. |
| 63 | Non-Afrikaans text: `olungu` repeats; no competent full semantic/syntax rating. |
| 65 | Non-Afrikaans text: a longer phrase repeats; source identity already corrupted. |
| 67 | Oral-sex phrase repeats and MIV remains absent. |
| 72 | Thousand pancakes becomes thousand nursery-school shops; donation/sale and names remain altered. |
| 75 | First-person pronoun restored locally; navigation and source condition remain incomplete. |

All 30 token-changed pairs are covered (29 decoded-text changes); the other 51 are token-identical. Improvements coexist with clear repetition and content failures. All outputs reached EOS before the128-position cap: repetition is not synonymous with failure to terminate. A progress message describing repetition continuing to the limit was premature and is corrected here. The automatic gate fails independently of these qualitative observations, so no subjective label is needed to justify stopping.
