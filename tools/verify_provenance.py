"""Check that the protocols and scripts recorded by each run are in the repository, unchanged.

Every hash in a run record (contract.json, or the AR audit's ARTIFACT_MANIFEST.json) whose key
names a protocol, plan or script is compared with the files at the `as-run` tag. A recorded hash
passes when some file at the tag has exactly that SHA-256 (as stored, or with LF line endings).
Hashes of models, data and sample banks are listed separately: they need the raw outputs, which
are not in the repository.

Usage:
  python tools/verify_provenance.py                 # run records committed in the repository
  python tools/verify_provenance.py --outputs DIR   # also the run records kept in a local outputs folder
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = 'as-run'
CODE_KEY = re.compile(r'protocol|plan|script|runner|entrypoint|support|analy[sz]|source|engine|driver', re.I)
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout


def tagged_hashes():
    """Map every SHA-256 (as stored, and LF-normalised) of text files at the tag to their paths."""
    out = {}
    for rel in git('ls-tree', '-r', '--name-only', TAG).decode('utf8').split('\n'):
        if not rel or rel.startswith('datasets/') or not re.search(r'\.(md|py|ps1|sh|slurm|json|txt|yml|yaml)$', rel):
            continue
        data = git('show', f'{TAG}:{rel}')
        for variant in (data, data.replace(b'\r\n', b'\n'), data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')):
            out.setdefault(hashlib.sha256(variant).hexdigest(), rel)
    return out


def records(extra):
    found = sorted((ROOT / 'docs/research').glob('*/results/contract.json'))
    found += [ROOT / 'docs/research/ar_validation_audit_2026-10-22/ARTIFACT_MANIFEST.json']
    if extra:
        found += sorted(Path(extra).glob('**/contract.json'))
    return found


def walk(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f'{prefix}.{k}' if prefix else k)
    elif isinstance(obj, str) and HEX64.match(obj):
        yield prefix, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', help='local outputs folder with further run records')
    args = ap.parse_args()
    try:
        git('rev-parse', TAG)
    except subprocess.CalledProcessError:
        sys.exit(f'tag {TAG} not found; run `git fetch --tags`')
    index = tagged_hashes()
    in_repo = set(git('ls-tree', '-r', '--name-only', TAG).decode('utf8').split('\n'))
    passed = failed = 0
    for path in records(args.outputs):
        rec = json.loads(path.read_text(encoding='utf-8-sig'))
        code, other = [], 0
        for key, value in walk(rec):
            if 'MANIFEST' in path.name:
                # manifest keys are file paths; check only protocols and scripts kept in the repository
                file = key.rsplit('.sha256', 1)[0]
                if file in in_repo and re.search(r'(PROTOCOL\.md|PLAN\.md|\.py|\.ps1|\.sh|\.slurm)$', file):
                    code.append((file, value))
                else:
                    other += 1
            elif CODE_KEY.search(key) and not key.startswith('source_code.'):
                code.append((key, value))
            else:
                other += 1
        if not code:
            continue
        label = path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)
        print(f'\n{label}')
        for key, value in code:
            hit = index.get(value)
            if hit:
                passed += 1
                print(f'  PASS  {key}: {hit}')
            else:
                failed += 1
                print(f'  ----  {key}: {value[:12]}… not found at tag {TAG}')
        if other:
            print(f'  ({other} further hashes cover models, data, outputs, logs or library snapshots not kept '
                  f'in the repository; they need the raw outputs and are not checked here)')
    print(f'\n{passed} recorded protocol/script hashes match files at tag {TAG}; {failed} do not.')


if __name__ == '__main__':
    main()
