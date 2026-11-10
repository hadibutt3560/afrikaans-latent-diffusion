"""Real architecture learning and resumed state equality with repaired engine."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'hotfix-20261010'),str(root),str(root/'vendor')]
import engine as e
import worker
from common import read,write

c=read(root/'config.json');hardware=e.setup(c)
folder=root/'local-checks'/'autocast-repair'
folder.mkdir(exist_ok=True,parents=True)
e.seed(42)
values=e.torch.nn.functional.normalize(e.torch.randn(8,32,64),dim=-1).numpy()*8
report={'hardware':hardware,'architectures':{}}
for name in c['models']:
    print('CHECK',name,flush=True)
    report['architectures'][name]=worker.resume_smoke(lambda:e.diffusion(c,name),values[:6],values[6:],c,'diffusion',folder/name,{'synthetic_repair_test':True})
    write(root/'verification'/'autocast-repair-gpu.json',report)
print('PASS: both full-size models learned; exact model/optimizer/EMA/RNG/scheduler resume',flush=True)
