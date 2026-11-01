"""Validate external cluster dependencies and weights before requesting training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.decoder_aware_cycle import digest, REVISION


def main():
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import snapshot_download
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args()
    expected={
        'outputs/nchlt-afrikaans/diffusion-50k-retry-20260920/model.pt':'09a888768393ee379690d28b164771d2fa818cbde7084b8dd16617744a3a604f',
        'outputs/afrikaans-24h-56531/best/model.pt':'123f4508e4aa26967c05a2ad413cebb8f2a2b90fc2420dab757704495dc27f5c'}
    for name,h in expected.items():
        if digest(a.base/name)!=h:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    expected_data={
        'train':'2af763380dd2818d40c58345428f42b8bebabbdcf71026c55bd2ea751601f3ab',
        'valid':'bdc360a9f2a65f5760830bd5bc1bc335d6ac066ec2087882d941c95255f9a91e'}
    for split,h in expected_data.items():
        if digest(Path('data/nchlt_afrikaans/processed/stage_5_7')/f'{split}.jsonl')!=h:
            raise ValueError(f'Dataset changed: {split}')
    config=AutoConfig.from_pretrained(a.base/'models/mt5-base',local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(a.base/'models/mt5-base',local_files_only=True)
    if config.decoder_start_token_id!=0 or tokenizer.eos_token_id!=1:
        raise ValueError('Unexpected tokenizer/config')
    vocab_hash=hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest()
    if vocab_hash!='5fa28c648562e2e68fbebee35fa1abff3e4bef8e1b1c223b9c003b373bba37d9':
        raise ValueError('Tokenizer vocabulary differs from the audited source')
    snapshot=Path(snapshot_download('ai-forever/mGPT',revision=REVISION,local_files_only=True))
    if not any(snapshot.glob('*.bin')) and not any(snapshot.glob('*.safetensors')):
        raise FileNotFoundError('Pinned mGPT weights absent; download before submission')
    gpu=None
    if not a.cpu_only:
        torch.cuda.init()
        value=torch.randn(64,64,device='cuda')
        if not torch.isfinite(value@value.T).all():
            raise RuntimeError('GPU arithmetic failed')
        gpu=dict(name=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
            vram=torch.cuda.get_device_properties(0).total_memory,torch=torch.__version__,cuda=torch.version.cuda)
    print(json.dumps(dict(status='passed',gpu=gpu,pinned_mgpt=str(snapshot),tokenizer_vocab_sha256=vocab_hash),indent=2))


if __name__=='__main__':
    main()
