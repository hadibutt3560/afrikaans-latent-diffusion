"""CPU API compatibility with the cluster's PyTorch/Transformers versions."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import transformers
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from transformers import MT5Config,T5Tokenizer,GPT2Config,GPT2LMHeadModel
from latent_models.t5_latent_model import MT5ForConditionalGenerationLatent
from transformers.modeling_outputs import BaseModelOutput
from common import write,rows


root=Path(__file__).parent;torch.set_num_threads(4)
assert transformers.__version__=='4.38.2'
assert torch.__version__.startswith('2.7.1')
c=MT5Config(vocab_size=64,d_model=32,d_kv=8,d_ff=64,num_layers=1,num_decoder_layers=1,num_heads=4,decoder_start_token_id=0,tie_word_embeddings=False)
m=MT5ForConditionalGenerationLatent(c,num_encoder_latents=4,num_decoder_latents=4,dim_ae=8,num_layers=1,l2_normalize_latents=True)
x=torch.tensor([[5,6,7,1]]);mask=torch.ones_like(x);z=m.get_diffusion_latent(m.get_encoder()(input_ids=x,attention_mask=mask),mask)
enc=BaseModelOutput(last_hidden_state=m.get_decoder_input(z));loss=m(labels=x,encoder_outputs=enc).loss;loss.backward()
with torch.no_grad():generated=m.generate(encoder_outputs=enc,max_length=6,num_beams=1)
tok=T5Tokenizer.from_pretrained(str(root/'assets/mt5-base'),local_files_only=True,legacy=True)
views=np.load(root/'tokens/train.npy',mmap_mode='r')
for i,row in enumerate(rows(root/'data/train.jsonl')):
    if i==128:break
    ids=tok.encode(row['text']);assert np.array_equal(views[i],ids+[0]*(192-len(ids)))
scorer=GPT2LMHeadModel(GPT2Config(vocab_size=64,n_embd=32,n_layer=1,n_head=4,n_positions=512))
score=float(scorer(input_ids=x,labels=x).loss.detach());assert np.isfinite(score)
write(root/'compatibility-check.json',dict(passed=True,torch=torch.__version__,transformers=transformers.__version__,
    tokenizer_parity_rows=128,ae_forward_backward_generate=True,scorer_forward=True,
    limitation='Windows CPU miniature API check; does not validate Linux Apptainer/CUDA 12.8 on Blackwell.'))
print('Pinned-version CPU API and real tokenizer parity passed')
