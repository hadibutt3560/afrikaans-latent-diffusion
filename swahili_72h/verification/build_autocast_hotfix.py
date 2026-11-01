"""Create a separately checksummed repair, retaining the original transfer bundle."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
dest=root/'hotfix-20261010'
dest.mkdir(exist_ok=True)
patch_id='diffusion-autocast-20261010-v1'
manifest={'id':patch_id,'reason':'PyTorch 2.7.1 cached no-grad self-conditioning weights suppress diffusion gradients',
          'scope':'Diffusion implementation repair before diffusion training; dataset, model, hyperparameters and test protocol unchanged', 'files':{}}

def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

for name in ('engine.py','worker.py','run.py','report.py'):
    original=(root/name).read_bytes();text=original.decode('utf8')
    if name=='engine.py':
        text=replace(text,"    with amp(c):\n        if kind=='ae':return ae_forward(model,values,ids)[0].loss\n",
            "    # Keep the AE path unchanged. A no-grad self-conditioning pass must not\n"
            "    # cache detached weight casts for the subsequent diffusion backward pass.\n"
            "    context=amp(c) if kind=='ae' else torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['precision']=='bf16',cache_enabled=False)\n"
            "    with context:\n        if kind=='ae':return ae_forward(model,values,ids)[0].loss\n")
        text=replace(text,"    params={n:p for n,p in model.named_parameters() if p.requires_grad}\n",
            "    if kind=='diffusion':contract['implementation']={'repair':'"+patch_id+"','engine_sha256':sha(Path(__file__))}\n"
            "    params={n:p for n,p in model.named_parameters() if p.requires_grad}\n")
        text=replace(text,"            opt.step()\n",
            "            if kind=='diffusion' and float(grad)==0:\n"
            "                raise FloatingPointError('Zero diffusion gradient: refusing a non-learning optimizer step')\n"
            "            opt.step()\n")
    elif name=='worker.py':
        text=replace(text,"        e.seed(713);model=factory()\n",
            "        e.seed(713);model=factory()\n"
            "        before=model.diffusion_model.output_proj.weight.detach().clone() if kind=='diffusion' else None\n")
        text=replace(text,"        e.train(model,train,valid,c,kind,folder/mode,713,contract,steps=n)\n        del model;e.clean()\n",
            "        e.train(model,train,valid,c,kind,folder/mode,713,contract,steps=n)\n"
            "        if before is not None:\n"
            "            if e.torch.equal(before,model.diffusion_model.output_proj.weight):raise AssertionError('Diffusion output weights did not learn')\n"
            "            del before\n"
            "        del model;e.clean()\n")
        text=replace(text,"result=dict(comparison=comparison,step=y['state']['step']);",
            "result=dict(comparison=comparison,step=y['state']['step'],learning_verified=(kind=='diffusion'));")
        text=replace(text,"    for name in c['models']:\n        e.seed(713);model=e.diffusion(c,name);small=copy.deepcopy(c);",
            "    learning_report={}\n"
            "    for name in c['models']:\n"
            "        print('REPAIRED DIFFUSION LEARNING/RESUME CHECK',name,flush=True)\n"
            "        learning_report[name]=resume_smoke(lambda:e.diffusion(c,name),train[:6],valid[:2],c,'diffusion',folder/('learning-'+name),a.contract)\n"
            "        write(a.out/'diffusion-learning-check.json',learning_report)\n"
            "        e.seed(713);model=e.diffusion(c,name);small=copy.deepcopy(c);")
    elif name=='run.py':
        text=replace(text,"    for name,digest in manifest['files'].items():\n        if sha(root/name)!=digest:raise ValueError('Bundle checksum mismatch: '+name)\n",
            "    repair=read(root/'runtime-patch.json') if (root/'runtime-patch.json').exists() else None\n"
            "    if repair and (repair['id']!='"+patch_id+"' or set(repair['files'])!={'engine.py','worker.py','run.py','report.py'}):raise ValueError('Unknown runtime repair')\n"
            "    for name,digest in manifest['files'].items():\n"
            "        expected=digest\n"
            "        if repair and name in repair['files']:\n"
            "            entry=repair['files'][name]\n"
            "            if entry['before']!=digest or sha(root/'runtime-patch-originals'/name)!=digest:raise ValueError('Repair baseline mismatch: '+name)\n"
            "            expected=entry['after']\n"
            "        if sha(root/name)!=expected:raise ValueError('Bundle checksum mismatch: '+name)\n")
    else:
        text=replace(text,"    if (out/'selection.json').exists():\n",
            "    if (out/'runtime-patch.json').exists():\n"
            "        results['runtime_patch']=read(out/'runtime-patch.json')\n"
            "        results['diffusion_learning_check']=read(out/'diffusion-learning-check.json') if (out/'diffusion-learning-check.json').exists() else None\n"
            "        results['limitations'].append('Initial diffusion resume smoke did not verify learning. A PyTorch 2.7.1 autocast-cache repair was installed before diffusion benchmarking/training; see runtime-patch.json and diffusion-learning-check.json for provenance and replacement checks.')\n"
            "    if (out/'selection.json').exists():\n")
    data=text.encode('utf8');(dest/name).write_bytes(data)
    compile(text,name,'exec')
    manifest['files'][name]={'before':hashlib.sha256(original).hexdigest(),'after':hashlib.sha256(data).hexdigest()}
(dest/'patch.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps(manifest,indent=2))
