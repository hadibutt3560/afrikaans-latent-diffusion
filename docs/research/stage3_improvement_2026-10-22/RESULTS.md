# Stage 3: removing the repeated-trigram restriction (finalised 23 October 2026)

Under the [protocol](PROTOCOL.md), the only change from Stage 1 was `no_repeat_ngram_size: 3 -> 0`, on the frozen autoencoder (job 56531, step 30500), the same 81 validation passages from 64 documents and the same latents (FP32, batch 1, beam 4, length 5–128, repetition penalty 1.2, length penalty 1). All 81 baseline sequences reproduced Stage 1 exactly. The change failed all three gates fixed in advance (at least +2 chrF, interval above zero, numeric recall down by no more than 2 points; recall fell 6.06 points), so the reserve and unconditional-transfer stages were not run and `no_repeat_ngram_size=3` is retained. Outputs (`pairs.jsonl`, `contract.json`, `analysis.json`, `checks.json`) are in `outputs/stage3-decoder-20261022/` in the research repository, not here. Runtime was 197.10 s after loading on an RTX 3060 (peak 2.38 GiB).

| Metric | Original restriction | Restriction removed |
|---|---:|---:|
| Mean sentence chrF | 67.7005 | 68.6190 |
| Exact reconstruction | 11/81 | 14/81 |
| Mean character SequenceMatcher | 0.77313 | 0.77442 |
| Source numeric strings retained | 21/33 (63.64%) | 19/33 (57.58%) |
| Numeric-string precision | 21/25 (84.00%) | 19/23 (82.61%) |
| Excess repeated word trigrams, summed over outputs | 0 | 80 |
| Excess repeated word fourgrams, summed over outputs | 0 | 57 |
| EOS reached | 81/81 | 81/81 |
| Output cap reached | 0/81 | 0/81 |

The chrF gain is 0.9185 points, paired document-bootstrap 95% interval [-0.1104, 2.0160] (10,000 replicates, seed 91428; SacreBLEU `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`). Thirty pairs changed in token sequence and 29 in text, all listed in [CHANGED_PAIRS.md](CHANGED_PAIRS.md): rows 11, 21 and 56 gain faithful copies, while rows 41, 48 and 57 repeat phrases (row 57 loses all clock times). Repeated n-grams are whitespace-word counts beyond the first occurrence, not a count of defective passages. Terminal projection was reviewed and deferred ([PROJECTION_ADMISSION.md](PROJECTION_ADMISSION.md)).
