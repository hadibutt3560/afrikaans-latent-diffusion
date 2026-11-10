"""Check the SHA256SUMS files in the repository against the files kept here.

Most hashes were recorded on Windows, so text files were hashed with CRLF line endings,
while git stores them with LF and a checkout may convert them. A file passes if its bytes
as checked out, or the same text with LF or CRLF line endings, match the recorded hash.
Entries for outputs not kept in the repository (model outputs, sample banks) are counted
as not checked.

Usage: python tools/check_output_hashes.py
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def forms(data):
    lf = data.replace(b'\r\n', b'\n')
    return {'as checked out': data, 'LF': lf, 'CRLF': lf.replace(b'\n', b'\r\n')}


def main():
    listed = subprocess.run(['git', '-C', str(ROOT), 'ls-files'], capture_output=True, text=True, check=True).stdout.split()
    sums = [f for f in listed if Path(f).name.startswith('SHA256SUMS')]
    ok = bad = missing = 0
    for s in sums:
        path = ROOT / s
        n_ok = n_bad = n_missing = 0
        for line in path.read_bytes().decode('utf8').splitlines():
            m = re.match(r'([0-9a-f]{64})\s+\*?(.+)', line.strip())
            if not m:
                continue
            digest, name = m.groups()
            target = path.parent / name
            if not target.is_file():
                n_missing += 1
                continue
            if any(hashlib.sha256(v).hexdigest() == digest for v in forms(target.read_bytes()).values()):
                n_ok += 1
            else:
                n_bad += 1
                print(f'  FAIL  {s}: {name}')
        print(f'{s}: {n_ok} match, {n_bad} fail, {n_missing} not in the repository')
        ok, bad, missing = ok + n_ok, bad + n_bad, missing + n_missing
    print(f'\n{ok} output hashes match, {bad} do not; {missing} cover outputs not kept in the repository.')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
