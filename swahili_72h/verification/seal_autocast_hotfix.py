"""Seal the verified small repair; never rebuild or overwrite the original bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path(__file__).resolve().parents[1];source=root/'hotfix-20261010'
gpu=json.loads((root/'verification/autocast-repair-gpu.json').read_text())
assert set(gpu['architectures'])=={'small','published'}
assert all(v['learning_verified'] and all(v['comparison'].values()) for v in gpu['architectures'].values())
base=json.loads((root/'bundle-manifest.json').read_text())
spec=json.loads((source/'patch.json').read_text())
for name,hashes in spec['files'].items():
    assert base['files'][name]==hashes['before']
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes['before']
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==hashes['after']
evidence=source/'verification';evidence.mkdir(exist_ok=True)
for name in ('autocast-repair-gpu.json','autocast-repair-gpu.log','autocast-pinned-cpu.log','autocast-installer-tests.log',
             'diagnose_diffusion_gradients.py','test_autocast_hotfix.py','hotfix_gpu_check.py','build_autocast_hotfix.py'):
    shutil.copyfile(root/'verification'/name,evidence/name)
(evidence/'VERIFIED.md').write_text('Local: PyTorch 2.7.1 CPU reproduces zero gradients with BF16 caching and restores gradients/weight updates without caching. RTX 3060 / PyTorch 2.11.0 verifies learning and exact 4-step versus 2+2-step model, optimizer, EMA, RNG and scheduler equality for both full-size architectures. Installer/idempotence/corruption/refusal and 10 orchestration tests passed. Cluster GPU repaired learning/resume checks remain pending and execute inside the existing benchmark stage.\n',encoding='utf8')
files=sorted(p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS')
checks={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(source/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in checks.items()),encoding='ascii')
archive=root/'transfer/swahili72-hotfix-20261010.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in files+[source/'SHA256SUMS']:
        info=tar.gettarinfo(str(p),arcname='hotfix-20261010/'+p.relative_to(source).as_posix())
        info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
        with p.open('rb') as f:tar.addfile(info,f)
with tarfile.open(archive) as tar:
    for name,expected in checks.items():
        assert hashlib.sha256(tar.extractfile('hotfix-20261010/'+name).read()).hexdigest()==expected
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'files':len(files)},indent=2))
