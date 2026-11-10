"""Standard-library persistence and protocol helpers (no model imports)."""
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def append(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def budget(elapsed, measured, runs, samples, test_records):
    """Seconds, conservative measured reserves; freeze before any test access."""
    # Caller supplies valid+test record count: conservative for either separate pass.
    selection=max(1800,2.5*runs*test_records*measured['validate'])
    evaluation=max(7200,2.5*(runs*(2*samples*measured['sample']+test_records*measured['validate'])+
        test_records*measured['encode']+128*measured['sample']+1024*measured['encode']))
    scoring=max(1800,2.5*((2*runs+2)*samples*measured['score']+300))
    reserve=max(4*3600,selection+evaluation+scoring)
    end = 72*3600 - 3600 - reserve
    return dict(evaluation_reserve_seconds=reserve,stage_reserves=dict(selection=selection,evaluate=evaluation,scoring=scoring),train_end_seconds=end,
                remaining_train_seconds=max(0, end-elapsed), shutdown_seconds=3600)
