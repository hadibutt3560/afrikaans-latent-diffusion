# Method decisions

Preregistered working method. Changes are dated and justified, never silent
replacements.

1. **English control.** ROCStories is the English implementation and reproduction
   control; it establishes pipeline fidelity, not the main contribution.
2. **Representation level.** The language autoencoder works on contextual
   encoder representations, not raw token IDs.
3. **Compression.** A Perceiver Resampler compresses variable token sequences to
   a fixed latent, initially length 32 and dimension 64.
4. **Reconstruction.** A Transformer reconstruction module maps the latent to
   continuous decoder inputs; the pretrained autoregressive decoder produces
   token logits.
5. **Autoencoder objective.** Token cross-entropy for the trainable bottleneck
   and reconstruction modules. The pretrained BART encoder/decoder stay frozen in
   the primary English reproduction.
6. **Noise schedule.** Continuous cosine,
   \(\bar{\alpha}(t)=\cos^2(\pi t/2)\), with \(t\in[0,1]\).
7. **Diffusion objective.** Unconditional ROCStories LD4LG reproduction: velocity
   prediction, L2 regression, cosine noise schedule.
8. **Sampling.** Published-style full configuration: 250-step DDPM; four-beam
   decoding.
9. **Token recovery.** No nearest-neighbour token rounding; the pretrained
   decoder maps reconstructed continuous features to logits over its fixed
   vocabulary.
10. **Afrikaans architecture.** First experiment: an Afrikaans-capable
    multilingual encoder-decoder such as
    [mT5](https://huggingface.co/google/mt5-base) with its existing SentencePiece
    tokenizer. Replacing the tokenizer before this baseline exists would break the
    pretrained decoder-vocabulary relationship.
11. **Afrikaans baseline order.** Evaluate the
    [Goldfish Afrikaans autoregressive model](https://huggingface.co/goldfish-models/afr_latn_1000mb),
    fine-tuning if justified, before considering a matched from-scratch
    autoregressive baseline.
12. **Independent scoring.** An independent multilingual scorer such as
    [mGPT](https://huggingface.co/ai-forever/mGPT) gives Afrikaans perplexity;
    scorer bias and tokenisation effects are documented.
13. **Success interpretation.** The proposed 20% perplexity threshold is
    secondary. Primary evidence: reconstruction fidelity, latent compression
    behaviour, diversity, memorisation, representation quality and interpretable
    failure modes.
14. **Metric separation.** English tokenisation and GPT-2-Large metrics serve the
    paper reproduction only and are not carried over unchanged to Afrikaans.
