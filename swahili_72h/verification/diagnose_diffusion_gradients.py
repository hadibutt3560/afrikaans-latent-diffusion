"""Read-only model diagnostic on synthetic data; does not touch run artifacts."""
import argparse
import json
import random
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
from model.diffusion_transformer import DiffusionTransformer
from gaussian_diffusion import GaussianDiffusion

p = argparse.ArgumentParser()
p.add_argument('--device', default='cpu')
a = p.parse_args()
torch.set_num_threads(4)
print(json.dumps({'torch': torch.__version__, 'device': a.device}), flush=True)
for enabled, cached in [(False, True), (True, True), (True, False)]:
    torch.manual_seed(713); random.seed(713)
    net = DiffusionTransformer(tx_dim=256, tx_depth=4, heads=4, latent_dim=64,
        max_seq_len=192, self_condition=True, scale_shift=True, dropout=.1, num_dense_connections=1)
    model = GaussianDiffusion(net, max_seq_len=32, objective='pred_v', loss_type='l2').to(a.device)
    model.l2_normalize = True
    opt = torch.optim.AdamW(model.parameters(), lr=.0001)
    z = torch.nn.functional.normalize(torch.randn(1,32,64,device=a.device),dim=-1)*8
    mask = torch.ones((1,32),device=a.device,dtype=torch.bool)
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        before = net.output_proj.weight.detach().clone()
        with torch.autocast(a.device, dtype=torch.bfloat16, enabled=enabled, cache_enabled=cached):
            loss = model(z, mask, None)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if enabled and cached and torch.__version__.startswith('2.7.1'):
            assert float(grad)==0 and net.output_proj.weight.grad is None
            assert torch.equal(before,net.output_proj.weight)
        else:
            assert float(grad)>0 and net.output_proj.weight.grad is not None
            assert not torch.equal(before,net.output_proj.weight)
        print(json.dumps({'bf16':enabled,'cache':cached,'step':step+1,'loss':float(loss.detach()),
            'gradient':float(grad),'head_gradient_none':net.output_proj.weight.grad is None,
            'head_delta':float((before-net.output_proj.weight).abs().max())}),flush=True)
