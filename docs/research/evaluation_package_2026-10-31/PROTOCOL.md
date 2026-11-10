# Language validity, scorer sensitivity, AR decoding sensitivity and a checklist judge: protocol

Fixed on 1 November 2026 before the run, this evaluation-only study (no training, no new diffusion sampling, no test-split or reserve access) asked whether generated outputs are identified as Afrikaans by GlotLID and how often they contain words an Afrikaans dictionary does not recognise; how much word 4-gram phrasing they share with the training split; whether three named perplexity comparisons point the same way under Goldfish as under mGPT; and how the AR baseline behaves over a fixed grid of sampling settings, with the diffusion comparison still judged against the 1.20 criterion on fresh outputs. Two gated pilots were included: MAUVE, reported only if the human median reached at least 0.90 (gate G-M), and a local checklist judge, used only if it detected at least 0.80 of planted faults per corruption type with a clean-text flag rate of at most 0.20 and at most 0.02 unparsed (gate G-J). Every measure was named as an automatic proxy: unrecognised tokens are not called invented words, overlap is not called memorisation, and Goldfish is an alternative, not independent, scorer.

The protocol also fixed the banks, seeds, intervals, outcome wording, budget gates and file hashes. The two appendices below are kept word for word because `run_package.py` reads the judge prompt from Appendix B and `overnight.py` records download hashes in Appendix C. The full protocol as fixed before the run is at the repository tag `as-run`: https://github.com/hadibutt3560/afrikaans-latent-diffusion/blob/as-run/docs/research/evaluation_package_2026-10-31/PROTOCOL.md.

## Appendix B. Checklist judge prompt (Part F)

The text between the lines is sent as the user message, with `{TEXT}` replaced by the output being rated, inside each candidate's own chat template. No system message is used.

---
You will read one passage that is meant to be written in Afrikaans, in the style of South African government text. Judge only the passage itself.

Answer each question with 1 (yes) or 0 (no):
J1: Is the text unreadable or incoherent as a whole?
J2: Does it contain a fragment repeated in a way a writer would not intend?
J3: Does it contain malformed Afrikaans words: wrong endings, broken compounds, or letter sequences that are not words?
J4: Is the word order or clause structure broken?
J5: Does it stop abruptly in the middle of a sentence or phrase?
J6: Is any substantial part of it in Dutch, English or another language rather than Afrikaans?
J7: Is it clearly out of place in South African government text?

Passage:
"""
{TEXT}
"""

Reply with exactly one line in this form and nothing else:
J1=0 J2=0 J3=0 J4=0 J5=0 J6=0 J7=0
---

Parsing: the first line of the reply matching `J1=[01] J2=[01] J3=[01] J4=[01] J5=[01] J6=[01] J7=[01]`, allowing extra spaces, is used. Anything else is unparsed.

## Appendix C. Downloaded resources (SHA-256)

Recorded from `E:/afrikaans-eval-20261031/pins.json` after download, before any run. Package versions in the new environment: `{'fasttext-turbo': '0.1.0', 'spylls': '0.1.7', 'mauve-text': '0.4.0', 'faiss-cpu': '1.15.1', 'sentence-transformers': '5.3.0', 'matplotlib': '3.11.2'}`.

| Resource | File | SHA-256 |
|---|---|---|
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `.gitattributes` | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `added_tokens.json` | `402f9f8b3242fc55fa48680ea341203a1e174f9bbaf32a2d41efb0939512ca11` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `config.json` | `b3d5b32f91e9311426e53a9600ebf322561198659fee0ff35d0174dd2257d46f` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `pytorch_model.bin` | `977ff3dc1ca47ed3b831cd9534d6afecb6d1416082d0665151cee980195f1989` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `README.md` | `297c2cd644a9185ba711e2cce59aee282f285fc9e635d4aa171db140e9daa2ad` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `special_tokens_map.json` | `4d6025468ad1902a29170ab995e005ad9ddb5e9cbf935dd0e40766a260d189c7` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `spiece.model` | `21c4b3da6315fbdd0f3d8631929efe31d74d7fd5c2694a84008bfb49318a0e02` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `tokenizer.json` | `6e025416e21598defef5b61085575c374200282dc64189582e3ccebefca3baa3` |
| goldfish-models/afr_latn_1000mb @ 27555fdf876b | `tokenizer_config.json` | `94e7853872729c0674041f829ec55c7157e13751e95c0ce8e76a439fbd8ab9c0` |
| cis-lmu/glotlid @ baa20d139151 | `model.bin` | `a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e` |
| sentence-transformers/LaBSE @ 836121a0533e | `1_Pooling/config.json` | `c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b` |
| sentence-transformers/LaBSE @ 836121a0533e | `2_Dense/config.json` | `02a7646de76be592655235523e09b5315c115bb770d91de87f21a9b0746723b5` |
| sentence-transformers/LaBSE @ 836121a0533e | `2_Dense/model.safetensors` | `f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77` |
| sentence-transformers/LaBSE @ 836121a0533e | `config.json` | `b6bd83d561db8d2ccb7f44dafcd945e39bea16dc2819072ffb10adb45608b029` |
| sentence-transformers/LaBSE @ 836121a0533e | `config_sentence_transformers.json` | `b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8` |
| sentence-transformers/LaBSE @ 836121a0533e | `model.safetensors` | `77d8e1f2dbab6eb5d3c261ce9d3dbf1e3c69e02938c95f934f94f42c22dfa31f` |
| sentence-transformers/LaBSE @ 836121a0533e | `modules.json` | `41251094dfd3e7ff81a85ba97b3f4788df47e4039f1bc5037575b43c028beb40` |
| sentence-transformers/LaBSE @ 836121a0533e | `sentence_bert_config.json` | `fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7` |
| sentence-transformers/LaBSE @ 836121a0533e | `special_tokens_map.json` | `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` |
| sentence-transformers/LaBSE @ 836121a0533e | `tokenizer.json` | `d09046654dc5151fc313613f16768b6ecc1e5e3bddbc9c8aca209a5f23fe3cd3` |
| sentence-transformers/LaBSE @ 836121a0533e | `tokenizer_config.json` | `9c5867ec63310dc2958daae8ccfabb226a2b6d317f407ade84d50315e7f9e80b` |
| sentence-transformers/LaBSE @ 836121a0533e | `vocab.txt` | `7453247caa2cf23116c1da332bec0cc1c9ef896b7803591a62ad2007b7a33c33` |
| ggml-org/gemma-4-12b-it-GGUF @ 202658970530 | `gemma-4-12B-it-Q4_0.gguf` | `3712b9bd32cae83a22f67ee7a4466d8d7a4f21646ac8a07d19bf9418e8767a70` |
| unsloth/Qwen3.5-9B-GGUF @ 3885219b6810 | `Qwen3.5-9B-Q4_K_M.gguf` | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| dict | `af_ZA.aff` | `fdfcdd04d700fc4ac94be63930167fa942ee5ffb471b52a6f4e0c68362dd63d2` |
| dict | `af_ZA.dic` | `8d806f90146059ea18b49c2352d188bd27dcc571a06815794facaa176aa5a5de` |
| dict | `README_af_ZA.txt` | `e00c86cb0a1c0499380770750f545b5d0308d6cb35d25014cef6891c2fe5f255` |
| ud | `nl_alpino-ud-test.conllu` | `20d0f718cdc73b51dbb2744386c9d4ecb1a9df6a076efa81b4c008ace3801fc0` |
| ud | `en_ewt-ud-test.conllu` | `fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a` |
| llama.cpp b11630 | `cudart-llama-bin-win-cuda-12.4-x64.zip` | `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |
| llama.cpp b11630 | `llama-b11630-bin-win-cuda-12.4-x64.zip` | `e260e0f3dbc93928205005b91a69e3ab037519ea7b0aac3c73d8faa38a5c69ae` |
