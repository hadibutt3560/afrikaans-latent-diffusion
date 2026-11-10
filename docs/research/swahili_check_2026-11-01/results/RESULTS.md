# Swahili check: results

Produced by `summarise.py` from the outputs of `run_check.py`, under PROTOCOL.md. All results are descriptive and on validation data.

## S1 reconstruction (200 passages)

| Arm | CE | Token accuracy | Exact | chrF | Numeral recall | Numeral precision | At cap |
|---|---|---|---|---|---|---|---|
| Correct | 0.521 | 0.868 | 0/200 | 74.8 | 0.54 | 0.67 | 0 |
| Shuffled | 6.662 | 0.262 | 0/200 | 24.3 | 0.00 | 0.01 | 0 |

Source-file bootstrap (2000 replicates, 35 files): exact rate [0.000, 0.000], chrF [72.9, 76.5], CE correct − shuffled -6.141 [-6.247, -6.030].

## S2 random latents (1,000)

Perplexity 135.3 against a diffusion validation mean of 19.26 (ratio 7.03). Fixed wording: **random latents score clearly worse than diffusion samples**. Mean words 80.0, distinct-2 0.743, empty 0, containing `<extra_id_` 0.

## S3 language identification (GlotLID top label, pooled)

| Bank | n | swh_Latn | Other Swahili | eng_Latn |
|---|---|---|---|---|
| real-valid | 256 | 1.000 | 0.000 | 0.000 |
| small-42-valid | 256 | 1.000 | 0.000 | 0.000 |
| small-43-valid | 256 | 1.000 | 0.000 | 0.000 |
| published-42-valid | 256 | 1.000 | 0.000 | 0.000 |
| published-43-valid | 256 | 1.000 | 0.000 | 0.000 |
| random-latents | 1000 | 0.568 | 0.010 | 0.001 |
| panel-sources | 200 | 0.990 | 0.000 | 0.000 |
| panel-reconstructions | 200 | 0.995 | 0.000 | 0.000 |

## S4 length and copying

| Bank | n | Mean words | mGPT PPL | distinct-2 | Exact copies | 4-gram overlap | High-overlap rows |
|---|---|---|---|---|---|---|---|
| real-valid | 256 | 47.8 | 13.26 | 0.840 | 0 | 0.275 | 3 |
| small-42-valid | 256 | 48.7 | 18.84 | 0.806 | 0 | 0.112 | 0 |
| small-43-valid | 256 | 48.7 | 18.56 | 0.807 | 0 | 0.110 | 0 |
| published-42-valid | 256 | 48.6 | 19.58 | 0.822 | 0 | 0.100 | 0 |
| published-43-valid | 256 | 48.7 | 20.04 | 0.827 | 0 | 0.104 | 0 |
| random-latents | 1000 | 80.0 | 135.30 | 0.743 | 0 | 0.005 | 0 |

The supplied splits have documented cross-split overlap, so the reference overlap is itself inflated.
