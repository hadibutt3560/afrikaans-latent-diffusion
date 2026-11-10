"""Copy audited helpers into a new experiment; never alter historical packages."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'full_decoder_12h_2026-10-24'
REPO = HERE.parents[2]
JOB = HERE / 'job'
JOB.mkdir(exist_ok=True)
sources = {}
def copy(source, destination, replacements=()):
    text = source.read_text(encoding='utf-8')
    sources[destination] = dict(source=str(source.relative_to(REPO)), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    for a,b in replacements:
        assert a in text, a
        text = text.replace(a,b)
    (JOB/destination).write_text(text, encoding='utf-8', newline='\n')

copy(REPO/'outputs/cluster-results/job-74177-review/protocol/engine.py', 'engine.py', (
    ('C2','D0'), ('F12','D1'),
    ("'RTX 8000' not in torch.cuda.get_device_name()", "'RTX PRO 6000 Blackwell' not in torch.cuda.get_device_name()"),
    ('checked RTX8000 on mscluster106 only', 'checked RTX PRO 6000 Blackwell on mscluster110 only'),
))
copy(OLD/'job/train_full.py', 'training_support.py', (
    ('C2','D0'), ('F12','D1'),
    ("blocks = range(10,12) if self.arm=='D0' else range(12)", 'blocks = range(12)'),
    ('plateau_stopping=False', 'plateau_stopping=True'),
))
copy(OLD/'job/transfer_eval.py', 'transfer_eval.py', (
    ('prediction=COUNT*(max(sampling)+2*max(decoding)+2*max(times))', 'prediction=COUNT*(max(sampling)+3*max(decoding)+3*max(times))'),
    ('reserve=max(5400.,prediction*2+1200.)', 'reserve=max(10800.,prediction*2+1200.)'),
    ('self.exp.guard()\n        with self.torch.no_grad():', 'self.exp.guard()\n        self.exp.ae.eval()\n        with self.torch.no_grad():'),
    ('del model;gc.collect();self.torch.cuda.empty_cache()\n        model,tok=self.scorer();anchors=[]', "del model;gc.collect();self.torch.cuda.empty_cache()\n        for arm in ('D0','D1'):\n            for i,z in enumerate(self.latents):\n                assert self.decode(z,SEED+i)['generated_ids']==probes[i]['generated_ids'],'Preflight identity decode failed: '+arm\n        model,tok=self.scorer();anchors=[]"),
    ("for arm in ('B0',candidate):", "for arm in (('B0',candidate) if smoke else ('B0','D0',candidate)):"),
    ('self.review(outputs,candidate,train)', "self.review(outputs,candidate,train)\n        control_result={}\n        if not smoke:\n            original_folder=self.folder\n            self.folder=original_folder/'control-comparison';self.folder.mkdir()\n            self.review(outputs,candidate,train,baseline='D0')\n            self.folder=original_folder\n            control_result=dict(candidate_over_control_intervals=paired_intervals(outputs['D0'],outputs[candidate],self.exp.np),\n                control_guardrails_passed=transfer_gate(summary['D0'],summary[candidate]))"),
    ('result=dict(status=\'completed\',smoke_only=smoke,', "result=dict(**control_result,status='completed',smoke_only=smoke,"),
    ('def review(self,outputs,candidate,train):', "def review(self,outputs,candidate,train,baseline='B0'):"),
    ("zip(outputs['B0'],outputs[candidate])):", 'zip(outputs[baseline],outputs[candidate])):'),
    ("A=candidate if flip else 'B0',B='B0' if flip else candidate", 'A=candidate if flip else baseline,B=baseline if flip else candidate'),
    ("(('B0',old),(candidate,new))", '((baseline,old),(candidate,new))'),
))
(JOB/'helper-provenance.json').write_text(json.dumps(sources,indent=2)+'\n', encoding='utf-8')
print('New helper sources prepared; historical files unchanged.')
