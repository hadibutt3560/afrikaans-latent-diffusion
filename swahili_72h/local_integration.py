"""Small real-model integration fixture. Uses VALIDATION rows only for all fake splits.

Never evaluates the actual Swahili held-out test set or tunes the scientific config.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common import read,write,rows
import engine as e
import worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--ae-smoke',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).parent;root=source/'local-fixture';out=root/'out'
    root.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    for folder in ('assets/mt5-base','assets/mgpt'):
        dest=root/folder;dest.mkdir(parents=True,exist_ok=True)
        for f in (source/folder).iterdir():
            if not (dest/f.name).exists():os.link(f,dest/f.name)
    c=copy.deepcopy(read(source/'config.json'));c.update(validation_examples=2,validation_repeats=1,samples_per_split=2)
    c['diffusion'].update(micro_batch=1,accumulation=1,validate_every=2,warmup=2,max_steps=4)
    c['ae']['micro_batch']=1
    write(root/'config.json',c);(root/'data').mkdir(exist_ok=True);(root/'tokens').mkdir(exist_ok=True)
    reference=list(__import__('itertools').islice(rows(source/'data/valid.jsonl'),8));tokens=np.load(source/'tokens/valid.npy',mmap_mode='r')[:8]
    for split in ('train','valid','test'):
        (root/'data'/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reference),encoding='utf8')
        np.save(root/'tokens'/f'{split}.npy',tokens)
    write(root/'data/audit.json',dict(record_counts=dict(train=8,valid=8,test=8),fixture='VALIDATION-only fixture, never real test'))
    write(root/'data/split_manifest.json',{'fixture':True})
    (out/'ae').mkdir(exist_ok=True)
    if not (out/'ae/best.pt').exists():os.link(a.ae_smoke,out/'ae/best.pt')
    args=SimpleNamespace(root=root,out=out,contract={'fixture':'VALIDATION-only local software checks'},run='small-42')
    e.setup(c)
    worker.cache(args,c)
    worker.diffusion_train(args,c)
    write(out/'plan.json',dict(admitted_runs=['small-42']))
    worker.selection(args,c)
    worker.evaluate(args,c)
    worker.scoring(args,c)
    write(out/'completion.json',dict(status='complete',scope='Software integration only; miniature model trained on validation fixtures. Not a research result.'))
    write(source/'local-integration.json',dict(passed=True,stages=['latent cache','diffusion train','full validation selection','fixture test evaluation','DDPM250 samples','mGPT scoring'],
          real_held_out_test_accessed=False))


if __name__=='__main__':main()
