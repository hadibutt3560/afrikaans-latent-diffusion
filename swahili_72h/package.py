"""Build a deterministic, checksummed self-contained transfer archive."""
import argparse
from pathlib import Path
import tarfile
from common import read,write,sha


def main():
    root=Path(__file__).parent.resolve();dest=root/'transfer';dest.mkdir(exist_ok=True)
    includes=['assets','data','tokens','vendor','wheelhouse','tests','verification']
    top=['common.py','audit.py','tokenize_data.py','engine.py','worker.py','run.py','report.py','metrics.py',
         'local_integration.py','compat_check.py','production_batch_check.py',
         'config.json','data-decision.json','source-provenance.json','assets-provenance.json','requirements.in','requirements.lock',
         'setup.sh','job.slurm','RUNBOOK.md','DATA_AUDIT.md','VERIFICATION.md','package.py']
    files=[root/n for n in top]
    for d in includes:
        files += [p for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                  not p.name.endswith(('.tmp','.partial','.pyc','-wal','-shm')) and p.name!='integrity-failure.json']
    manifest={p.relative_to(root).as_posix():sha(p) for p in sorted(files)}
    write(root/'bundle-manifest.json',dict(schema=1,files=manifest))
    archive=dest/'swahili72-transfer.tar';temp=archive.with_suffix('.tar.tmp')
    with tarfile.open(temp,'w') as tar:
        for p in sorted(files+[root/'bundle-manifest.json']):
            info=tar.gettarinfo(str(p),arcname='package/'+p.relative_to(root).as_posix());info.uid=info.gid=0
            info.uname=info.gname='';info.mtime=0;info.mode=0o755 if p.suffix in ('.sh','.slurm') else 0o644
            with p.open('rb') as f:tar.addfile(info,f)
    temp.replace(archive);digest=sha(archive)
    (dest/'swahili72-transfer.tar.sha256').write_text(digest+'  swahili72-transfer.tar\n',encoding='ascii')
    # Read archive contents and verify every byte against the manifest after writing.
    import hashlib
    with tarfile.open(archive,'r') as tar:
        for name,expected in manifest.items():
            h=hashlib.sha256()
            with tar.extractfile('package/'+name) as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=expected:raise RuntimeError('Archive verification failed: '+name)
    write(dest/'bundle-verification.json',dict(passed=True,files=len(manifest),bytes=archive.stat().st_size,sha256=digest))
    print(archive,digest,flush=True)


if __name__=='__main__':main()
