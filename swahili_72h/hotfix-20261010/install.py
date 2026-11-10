"""Install the bounded, checksummed repair using only Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def atomic(p, data):
    tmp=p.with_name(p.name+'.repair-tmp')
    with tmp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def install(root,out,source):
    spec=read(source/'patch.json')
    assert spec['id']=='diffusion-autocast-20261010-v1'
    assert set(spec['files'])=={'engine.py','worker.py','run.py','report.py'}
    manifest=read(root/'bundle-manifest.json')
    existing=root/'runtime-patch.json'
    if existing.exists():
        receipt=read(existing)
        assert receipt['files']==spec['files'] and receipt['id']==spec['id'],'Different repair already installed'
        for name,hashes in spec['files'].items():
            assert digest(root/name)==hashes['after'],name
        assert (out/'runtime-patch.json').exists(),'Missing results receipt'
        print('Repair already installed and verified; no changes made.')
        return
    stages=read(out/'stages.json')
    assert stages and set(stages)<={'smoke','ae_train','ae_gate','cache'},'Benchmark or later stage already started; do not hot-patch this run'
    assert stages.get('smoke',{}).get('complete'),'Original smoke must have finished'
    assert not any((out/name).exists() for name in ('benchmark','benchmark.json','plan.json','runs','selection.json','ae-test.json')),'Downstream work already exists; stop and review'
    if not stages.get('ae_train',{}).get('complete'):
        assert stages['ae_train']['deadline']-time.time()>300,'AE transition too close; obtain fresh status first'
    for name,hashes in spec['files'].items():
        assert manifest['files'][name]==hashes['before'],'Unexpected baseline manifest: '+name
        assert digest(root/name)==hashes['before'],'Original changed: '+name
        assert digest(source/name)==hashes['after'],'Damaged repair payload: '+name
        compile((source/name).read_text(encoding='utf8'),name,'exec')
    original=root/'runtime-patch-originals';original.mkdir(exist_ok=True)
    evidence=out/'provenance'/'runtime-patch';evidence.mkdir(parents=True,exist_ok=True)
    for name in spec['files']:
        backup=original/name
        if backup.exists():assert digest(backup)==spec['files'][name]['before'],'Backup mismatch'
        else:shutil.copyfile(root/name,backup)
        shutil.copyfile(source/name,evidence/name)
        shutil.copyfile(backup,evidence/('original-'+name))
    for name in ('patch.json','install.py','README.md'):
        shutil.copyfile(source/name,evidence/name)
    if (source/'verification').exists():
        checks=evidence/'verification';checks.mkdir(exist_ok=True)
        for entry in (source/'verification').iterdir():
            if entry.is_file():shutil.copyfile(entry,checks/entry.name)
    receipt=dict(spec,installed_epoch=time.time(),baseline_manifest_sha256=digest(root/'bundle-manifest.json'),
        stages_at_install=stages,installer_sha256=digest(source/'install.py'),
        note='Existing AE process retains its loaded code. Subsequent workers use the repair. Original diffusion smoke is superseded by benchmark learning/resume checks. No model test results were consulted.')
    payload=(json.dumps(receipt,indent=2)+'\n').encode()
    changed=[]
    try:
        for name in ('run.py','report.py','worker.py','engine.py'):
            atomic(root/name,(source/name).read_bytes());changed.append(name)
        atomic(out/'runtime-patch.json',payload)
        atomic(existing,payload)
    except BaseException:
        for name in reversed(changed):atomic(root/name,(original/name).read_bytes())
        raise
    for name,hashes in spec['files'].items():assert digest(root/name)==hashes['after']
    print('INSTALLED AND VERIFIED: '+spec['id'])
    print('Keep the current allocation running. No sbatch or restart is required.')
    print('GPU learning/resume results will appear in '+str(out/'diffusion-learning-check.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();install(a.root.resolve(),a.out.resolve(),Path(__file__).resolve().parent)
