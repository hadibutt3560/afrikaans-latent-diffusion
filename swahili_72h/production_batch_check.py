"""Local production microbatch/accumulation and published DDPM250 smoke, train data only."""
import gc
from pathlib import Path
import time
import numpy as np
import engine as e
from common import read,write


root=Path(__file__).parent;c=read(root/'config.json');hardware=e.setup(c);report={'hardware':hardware,'checks':{}}
for kind,name in [('ae','ae'),('diffusion','small'),('diffusion','published')]:
    e.seed(42);model=e.ae_model(root,c) if kind=='ae' else e.diffusion(c,name)
    spec=c[kind];params=[p for p in model.parameters() if p.requires_grad]
    opt=e.torch.optim.AdamW(params,lr=spec['lr']);ema=[p.detach().clone() for p in params] if spec['ema'] else []
    if kind=='ae':values=np.load(root/'tokens/train.npy',mmap_mode='r');model.get_encoder().eval()
    else:values=e.torch.nn.functional.normalize(e.torch.randn(64,32,64),dim=-1).numpy()*8
    e.torch.cuda.reset_peak_memory_stats();tick=time.monotonic();total=0.;n=spec['micro_batch']*spec['accumulation']
    for i in range(0,n,spec['micro_batch']):
        v=e.loss(model,values,np.arange(i,i+spec['micro_batch']),c,kind)/spec['accumulation'];assert e.torch.isfinite(v)
        v.backward();total+=float(v.detach())
    grad=e.torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
    assert all(e.torch.isfinite(p).all() for p in params)
    if ema:
        with e.torch.no_grad():
            for target,p in zip(ema,params):target.lerp_(p,1-spec['ema'])
    e.torch.cuda.synchronize();report['checks'][name]=dict(batch=n,micro_batch=spec['micro_batch'],accumulation=spec['accumulation'],
        loss=total,gradient=float(grad),seconds=time.monotonic()-tick,peak_vram=e.torch.cuda.max_memory_allocated())
    del opt,params,ema;e.clean()
    if name=='published':
        ae=e.ae_model(root,c).eval().requires_grad_(False)
        sample=e.generate(model,ae,e.tokenizer(root),c,1,891273)
        report['published_DDPM250_beam4']=dict(passed=True,output_tokens=sample[0]['tokens']);del ae
    del model;e.clean()
write(root/'production-batch-check.json',report);print('Production batches and published DDPM250 passed',flush=True)
