"""Local-only verification: load the same FP32 scorer directly on GPU.

The Windows Transformers5 CPU-first loader crashed natively. This changes only
loading placement for local verification; the cluster's proven Transformers4.38
loader and scientific settings are unchanged. Four historical anchors must pass.
"""
from pathlib import Path
import os
import sys

REPO=Path(__file__).resolve().parents[3]
PACKAGE=REPO/'outputs/cluster-transfer/afrikaans-reconstruction-joint-20261025'
sys.path.insert(0,str(PACKAGE))
os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
import train_joint
import transfer_eval

def scorer(self):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers.modeling_utils as modeling_utils
    # Local Transformers5 allocator warmup duplicates a model-sized allocation.
    # Disable only that loading optimization; weights and FP32 scoring unchanged.
    modeling_utils.caching_allocator_warmup=lambda *args,**kwargs:None
    path=PACKAGE/'evaluation/mgpt'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        torch_dtype=self.torch.float32,device_map={'':'cuda'}).eval().requires_grad_(False)
    return model,tokenizer

transfer_eval.Transfer.scorer=scorer
print('LOCAL_VERIFICATION_ONLY: same pinned FP32 scorer, direct GPU loading; cluster source unchanged.',flush=True)
train_joint.main()
