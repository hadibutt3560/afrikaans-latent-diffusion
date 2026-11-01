# AI-assisted reconstruction inspection

This is an AI-assisted comparison of source and correct-latent output, not a fluent-human assessment, blinded evaluation or measured grammar score. Every diagnostic row is indexed in manifest order. The notes identify visible preservation failures; silence about a category does not certify it. Original references can themselves be fragments, English text, boilerplate or misspellings. Faithful copying and grammatical correction are different tasks. No percentage of linguistic adequacy is inferred from these notes.

Full source, tokenized view and both reconstructions are in `ALL_RECONSTRUCTIONS.md`; exact IDs, numeric scores and donor IDs are in `outputs/reconstruction-audit-20261022/examples.jsonl`.

| Row | Observed preservation or discrepancy |
|---|---|
| 0 | Department becomes commission; regulation becomes payment; added taxi-office clauses. |
| 1 | Service-delivery area replaced by website fragment. |
| 2 | Nearly faithful; source spelling `ingevloge` corrected to `ingevolge`. |
| 3 | English contact-person fragment replaced by `Tel na`. |
| 4 | Department/name and scholarship conditions changed; 20 omitted; men-only condition changed to men and women. |
| 5 | University list altered; UNESCO introduced; Stellenbosch omitted. |
| 6 | Ministry changed to minister; remainder preserved. |
| 7 | Road rage changed to road construction. |
| 8 | Ministry changed to minister; remainder preserved. |
| 9 | Appeal/application alternatives compressed; `teen, of aansoek` omitted. |
| 10 | Collection/recovery concept added to a source fragment. |
| 11 | Apostrophe changed; wording otherwise preserved. |
| 12 | Exact reconstruction. |
| 13 | Thabo omitted, UOWP changed to OPS, Cape Town added. |
| 14 | Early-childhood-development phrase replaced by unrelated service conditions. |
| 15 | Person changes from `ek` to `u` in final clause; source wording and apostrophe altered. |
| 16 | English contact-person fragment replaced by office fragment. |
| 17 | English source: place names/region altered and Health repeated. |
| 18 | Exact short fragment. |
| 19 | Competitor names and team composition changed; medal clause repeated. |
| 20 | Name, age/weight, dates, venue and competition details altered or omitted; World Bank added. |
| 21 | Second PHONE placeholder changed to EMAIL. |
| 22 | Import/export control replaced by unrelated civic-centre fragment. |
| 23 | Firearms changed to regulation; Act 60 changed to Act 1. |
| 24 | Exact source fragment. |
| 25 | All hours changed to all countries; markup-like suffix added. |
| 26 | Human society changed to affordable/sustainable society; quality wording altered. |
| 27 | Per year changed to per month; first `nie` omitted from double negation. |
| 28 | Ministry changed to minister; remainder preserved. |
| 29 | Mother changed to child; pronouns and options clause changed; source HTML entity not faithfully retained. |
| 30 | Mixed-language navigation and question heavily shortened/altered; family/friends omitted. |
| 31 | Testicular pain replaced by unrelated wine/negation clause. |
| 32 | Daughter omitted from question; Grade19 introduced in navigation. |
| 33 | Someone changed to I; employment clause altered. |
| 34 | Non-Afrikaans source and output; extensive lexical substitutions visible. Language competence insufficient for a syntax judgement. |
| 35 | Compound spacing and preposition changed; most content retained. |
| 36 | Exact address fragment. |
| 37 | Exact date/volume fragment. |
| 38 | Exact English file-metadata fragment. |
| 39 | Conjunction changed to colon; most wording retained. |
| 40 | Registration/training requirements broken into altered clauses; added successor wording. |
| 41 | Mixed-language navigation/question replaced by different names/fragments. |
| 42 | Extra file/type wording inserted in English metadata. |
| 43 | Office/financial-statement text added to English metadata. |
| 44 | Training omitted; spelling altered. |
| 45 | Tour-guide database replaced by Western Cape centre; purpose clause omitted. |
| 46 | Exact short question. |
| 47 | Curriculum-delivery phrase altered into web-plan wording; inflections changed. |
| 48 | Counselling-before-testing question and navigation heavily altered. |
| 49 | Laws affecting business become laws encouraging it; businesses become small farmers. |
| 50 | Named hospitals and hearing aids replaced by homepage/information-centre wording. |
| 51 | Company names, chickens, vaccine and contact institution altered/omitted; Dublin added. |
| 52 | Director title changed; contingency plans altered; disease-control clause omitted. |
| 53 | Exact English sentence. |
| 54 | Safe-installation question replaced by community/work question. |
| 55 | Nonprofit organisation phrase corrupted and abbreviation lost. |
| 56 | Trauma/psychological assessment changed to attack/illness assessment. |
| 57 | Opening and closing times lost; weekdays repeated and new 08:30 time introduced. |
| 58 | Exact document-requirement sentence. |
| 59 | Category ordering replaced by data-infrastructure addition. |
| 60 | Specialist functions and procurement-policy wording altered. |
| 61 | English contact-person fragment corrupted. |
| 62 | EMDC changed to EOI and Winelands to Winslow. |
| 63 | Non-Afrikaans source: ministry/contact wording changes visible, PHONE preserved; no competent syntax rating. |
| 64 | Nutrition/immune-response question replaced by generic goals question; navigation changes. |
| 65 | Non-Afrikaans source: FLOWERLINE and surrounding words substantially altered; no competent syntax rating. |
| 66 | Non-Afrikaans source replaced by unrelated Afrikaans-like learning/criteria sentence. |
| 67 | MIV omitted from the prevention question. |
| 68 | Partner's mouth changed to environment; negative construction retained but scope/content altered. |
| 69 | Exact English sentence. |
| 70 | Identification/documentation and mirrors/seatbelts altered or combined. |
| 71 | More-than-fifty outcome omitted; collection changed to switching on; other clause changes. |
| 72 | Station/event names altered; donation and pancake sale changed to sale of articles; Kansa changed to Jane. |
| 73 | `daarna` repeated; remainder preserved. |
| 74 | Singular/plural change in notices; negation preserved. |
| 75 | Mixed-language navigation shortened/altered; child-interest phrase and trailing condition omitted. |
| 76 | Address and 391 replaced by government-comparison fragment. |
| 77 | Special emphasis changed to necessary emphasis; communities changed to groups. |
| 78 | Exact sentence. |
| 79 | Short address/name fragment replaced. |
| 80 | Personalised numberplates changed to manufactured numberplates. |

## Interpretation

The inspection covers all81 correct-latent outputs. There are near-copies, exact copies, entity changes, numeric changes, omissions, additions and altered clause relationships. Row27 is a clear annual/monthly substitution and negation omission; row20 alters multiple names/dates; row1 fails on a short source. Mixed-language and metadata inputs also occur. These observations support a reconstruction-fidelity limitation under the fixed decoder; they do not assign its cause uniquely to compression, encoder or decoder. The numeric results remain the primary reproducible evidence. Fluent-human adequacy assessment is not completed by this AI inspection.
