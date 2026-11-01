# Method decisions

These decisions are the current preregistered working method. Changes should be
dated and justified rather than silently replacing them.

1. **English control.** ROCStories is the English implementation and reproduction
   control. English results establish pipeline fidelity; they are not the main
   research contribution.
2. **Representation level.** The language autoencoder operates on contextual
   encoder representations, not directly on raw token IDs.
3. **Compression.** A Perceiver Resampler compresses variable token sequences to
   a fixed latent, initially length 32 and dimension 64.
4. **Reconstruction.** A Transformer reconstruction module maps the compressed
   latent to continuous decoder inputs. The pretrained autoregressive decoder then
   produces token logits.
5. **Autoencoder objective.** The trainable bottleneck and reconstruction modules
   use token cross-entropy. The pretrained BART encoder/decoder remain frozen in
   the primary English reproduction.
6. **Noise schedule.** The standard continuous cosine schedule is
   \(\bar{\alpha}(t)=\cos^2(\pi t/2)\), with \(t\in[0,1]\).
7. **Diffusion objective.** The unconditional ROCStories LD4LG reproduction uses
   velocity prediction, L2 regression, and cosine noise scheduling.
8. **Sampling.** The published-style full configuration uses 250-step DDPM
   sampling. The decoder uses four-beam generation.
9. **Token recovery.** No nearest-neighbour token rounding is required. The
   pretrained language decoder converts reconstructed continuous features to
   logits over its fixed vocabulary.
10. **Afrikaans architecture.** The first Afrikaans experiment should use an
    Afrikaans-capable multilingual encoder-decoder such as
    [mT5](https://huggingface.co/google/mt5-base), including its existing
    SentencePiece tokenizer. Replacing the tokenizer before establishing this
    baseline would break the pretrained decoder-vocabulary relationship.
11. **Afrikaans baseline order.** Evaluate and then, if justified, fine-tune the
    [Goldfish Afrikaans autoregressive model](https://huggingface.co/goldfish-models/afr_latn_1000mb)
    before spending time on a matched from-scratch autoregressive baseline.
12. **Independent scoring.** Use an independent multilingual scorer such as
    [mGPT](https://huggingface.co/ai-forever/mGPT) for Afrikaans perplexity, while
    explicitly documenting scorer bias and tokenisation effects.
13. **Success interpretation.** The proposed 20% perplexity threshold is
    secondary. Reconstruction fidelity, latent compression behaviour, diversity,
    memorisation, representation quality, and interpretable failure modes are
    primary evidence.
14. **Metric separation.** English tokenisation and GPT-2-Large metrics are for
    paper reproduction. They must not be copied unchanged to Afrikaans.
