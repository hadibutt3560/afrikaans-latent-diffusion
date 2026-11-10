"""Read a repository file exactly as it was at the `as-run` tag.

The protocols and scripts used by the recorded runs are frozen at the git tag `as-run`; their
SHA-256 values are stored in each run's contract.json. Later commits may tidy the wording of
these files, so checks against recorded hashes read the tagged version instead of the working
copy. Requires a git checkout that includes the tag (`git fetch --tags`).
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = 'as-run'


def as_run_bytes(path):
    """Bytes of `path` (absolute, or relative to the repository root) at the `as-run` tag."""
    rel = Path(path).resolve().relative_to(ROOT).as_posix() if Path(path).is_absolute() else Path(path).as_posix()
    try:
        return subprocess.run(['git', 'show', f'{TAG}:{rel}'], cwd=ROOT, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f'cannot read {rel} at tag {TAG}; clone the repository with git and run `git fetch --tags`') from e


def as_run_sha256(path, lf=True):
    """SHA-256 of `path` at the `as-run` tag; with lf=True, line endings normalised to LF."""
    data = as_run_bytes(path)
    if lf:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def as_run_matches(path, expected):
    """True if `path` at the `as-run` tag has SHA-256 `expected` with LF or CRLF line endings.

    Runs recorded the hash of the file as it was on the machine that ran them, so either
    line-ending convention may appear in a run record.
    """
    lf = as_run_bytes(path).replace(b'\r\n', b'\n')
    return expected in {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest()}


if __name__ == '__main__':
    # Command-line check used by non-Python verifiers: python tools/as_run.py check PATH SHA256
    import sys
    if len(sys.argv) != 4 or sys.argv[1] != 'check':
        sys.exit('usage: python tools/as_run.py check PATH SHA256')
    ok = as_run_matches(Path(sys.argv[2]).resolve(), sys.argv[3].lower())
    print(('matches' if ok else 'DIFFERS from') + f' the run record at tag {TAG}: {sys.argv[2]}')
    sys.exit(0 if ok else 1)
