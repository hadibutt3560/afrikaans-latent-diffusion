# Screen protocol sample reading (Task B)

- **Reader:** Claude Opus 5.5 (`claude-opus-5-5`), on 7 November 2026.
- **Status:** an AI-assisted reading, not human evaluation. The reader knew the study aims.
- **Samples:** test outputs at indices 0, 25, 50, 75 and 100 of seed 42 for each retrained denoiser. These follow the fixed rule in `../ae_screen_2026-10-27/PROTOCOL.md` §7.
- **Source:** `outputs/cluster-results/ae-diffusion-77176/screen/diffusion/runs/{small,published}-seed42/test-selected-samples.json` in the research workspace.

| Size | Index (seed) | Text | Afrikaans? | Grammatical? | Coherent? | Note |
|---|---|---|---|---|---|---|
| small | 0 (120000) | Wanneer ons sal handig aan aloer plantslimmere en (a), waarom; so ja die tersaak van een mense onderkry na die neem van drone. | Yes | No | No | Non-words (aloer, plantslimmere); no main clause |
| small | 25 (120025) | Moeding klasse gebruik op die grondsektor. | Yes | No | Partly | Non-word (Moeding); fragment |
| small | 50 (120050) | Die Agentsies wat reeds gemaak word, aansoek om goedgee beleggings wil, die applikaind bedryf as 'n professionele prys Troulbank en neem vir hierdie verslag gehandel? | Yes | No | No | Non-words (goedgee, applikaind, Troulbank); government-form register |
| small | 75 (120075) | Toool Die Besoek wat die aansoekvorm dien is? | Yes | No | No | Non-word (Toool) |
| small | 100 (120100) | Die keule as 'n werker sal die voorsien sentrum binne die diensinspekteur en die gereedskapte verdienste van indien hierdie regering gereik word as die beplanning van verbinding. | Yes | No | No | Non-words (gereedskapte, gereik); no coherent clause |
| published | 0 (120000) | Vandag, hierdie ongeluk sal erkenning gee aan doelleerders in die pad tot veld vir die pleeglig. Hulle behoort 10 liter by te staan na die neem van diarree. | Yes | Partly | No | Sentence frames are grammatical; content is absurd; non-word (pleeglig) |
| published | 25 (120025) | Koerd moet gebruik word indien die gewas gebruik is. Kortte sal nie stoor vasstel nie en as implement uitgesluit word. | Yes | Partly | Partly | First sentence is grammatical; non-words (Koerd, Kortte) |
| published | 50 (120050) | As inligting met enige wat reeds gebeur wil word, is die CTFS Park aanlyn as me Mafikhilomo, wens vir meer inligting: <PHONE>. | Yes | No | Partly | Follows the corpus's contact-detail template |
| published | 75 (120075) | Toon dat die besoek en die siekte ge�dentifiseer het. | Yes | Yes | Partly | Decoding error in geïdentifiseer (replacement character) |
| published | 100 (120100) | Die komitee kan CD's vestig teenoor die bestuursraamwerk en die steunverskaffers gedurende hierdie jaar na administrasie gee tot die begin van 21 September 2009. | Yes | Partly | Partly | Reads like government text; the verb phrases do not fit |

## Summary

- **Afrikaans:** all 10 outputs read as Afrikaans.
- **Small denoiser:** all 5 small outputs contain non-words, and none is grammatical.
- **Published denoiser:** the larger denoiser's outputs are more often grammatical at sentence level. Two contain non-words and one has a decoding error.
- **Coherence:** no output is coherent from start to finish.
- **Scope:** this reading is descriptive only. It is not used in any criterion.
