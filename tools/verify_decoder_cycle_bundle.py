"""Verify every packaged byte against its manifest without extracting files."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def main():
    archive=Path(sys.argv[1])
    prefix='afrikaans-decoder-aware-20260930/'
    with tarfile.open(archive,'r:gz') as tar:
        names=tar.getnames()
        if len(names)!=len(set(names)) or any(not n.startswith(prefix) or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe or duplicate archive member')
        manifest=tar.extractfile(prefix+'bundle-manifest.sha256').read().decode().splitlines()
        for line in manifest:
            expected,name=line.split('  ',1)
            actual=hashlib.sha256(tar.extractfile(prefix+name).read()).hexdigest()
            if actual!=expected:
                raise ValueError(name)
    print(json.dumps(dict(status='passed',archive=str(archive),files_verified=len(manifest),
        bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    main()
