# Dated note, 1 November 2026: primary contrasts not estimable; C2 reported as post hoc

Written after the Phase A run (`outputs/latent-controls-20261030`, all gates passed) and its analysis. The approved protocol is unchanged; this note records how its outcome is reported.

## Outcome under the protocol

- **Both primary contrasts (small − C3 and large − C3) are insufficient support.** C3 placed 2, 278, 666 and 54 of its 1,000 outputs in the four length bands, so the 1–10-word band is below the 20-row minimum (§5). Under §7 the permitted wording is: "Length-standardised comparison not reported: C3 has fewer than 20 outputs in band 1–10." C3's raw perplexity (67.86) is reported without a classification.
- **Why:** recombining slots from independent passages almost never yields a short text. This is a property of the control, not a failure of the run, and it was not foreseen when C3 was chosen as the primary control.

## What is reported instead, and how it is labelled

The exploratory contrasts (§6, 95%, t = 2.7764) are reported as **exploratory, not confirmatory**:

| Contrast | Difference | 95% interval | Class |
|---|---|---|---|
| small − C1 | −36.35 | [−53.09, −19.61] | lower |
| large − C1 | −48.06 | [−64.69, −31.43] | lower |
| small − C2 | +7.72 | [+3.74, +11.71] | higher |
| large − C2 | −3.99 | [−7.48, −0.49] | lower |

- **C2, the Gaussian fitted to training latents, is the informative comparison.** Raw perplexity does not separate any condition from diffusion (C1 raw 50.36 against diffusion means of 51.94 and 49.60). Any text giving C2 a central role must say that this emphasis was chosen **after** seeing the results.
- C2's outputs are less diverse than diffusion (distinct-2 0.629 against 0.670 and 0.699; unigram entropy 8.80 against 9.26 and 9.46 bits), so part of its lower perplexity reflects blander text.

## Not done

No confirmatory replacement for C3 is defined here. A future comparison against C2, for example on the new autoencoder, needs its own approved protocol with C2 named as primary before it is run.
