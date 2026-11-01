"""Apply the author-approved 26 October timing amendment; never submit a job."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd().resolve()
assert root.name=='afrikaans-reconstruction-joint-20261025',root
assert not (root/'submitted-job-id.txt').exists(),'Submission already recorded; do not change a submitted package'
backup=root/'timing-original-20261026'
assert not backup.exists(),'Amendment already applied or partially applied; inspect before retrying'
manifest_path=root/'bundle-manifest.sha256'
original=manifest_path.read_bytes()
entries={name:sha for sha,name in (line.split('  ',1) for line in original.decode().splitlines())}
changed={}
for name in ('submit.sh','joint.slurm','PLAN.md'):
    raw=(root/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==entries[name],f'Original file mismatch: {name}'
    changed[name]=raw.decode()
old='2026-10-26T06:00:00+02:00'
assert changed['submit.sh'].count(old)==1
changed['submit.sh']=changed['submit.sh'].replace(old,'2026-10-26T15:00:00+02:00')
old_line='"$PY" -u train_joint.py --ae "$AE" --model-cache "$MODEL_CACHE" --output "$OUT" &'
assert changed['joint.slurm'].count(old_line)==1
assert '#SBATCH --time=12:00:00' in changed['joint.slurm']
changed['joint.slurm']=changed['joint.slurm'].replace(old_line,
    '"$PY" -u train_joint.py --ae "$AE" --model-cache "$MODEL_CACHE" '
    '--start-before "2026-10-26T15:00:00+02:00" '
    '--deadline "2026-10-27T03:00:00+02:00" --output "$OUT" &')
changed['PLAN.md']+='''

## Timing amendment authorized 26 October 2026

This paragraph supersedes only the original start/finish dates above. User requested
the full12-hour batch after the original start window expired. Latest actual job
start is now26October15:00SAST; absolute finish27October03:00SAST, leaving at least
21hours before submission at the end of27October. Slurm ceiling12h, training/setup
phase<=8h, evaluation reserve>=3h, export margin, paired workload, admission,
stopping rules and all scientific gates are unchanged. This is a maximum, not a
minimum duration. Queued jobs starting too late must abort without training.
No result motivated this timing-only amendment; the experiment has not started.
Original files and manifest are preserved in timing-original-20261026/.
'''
changed['PLAN.md']+='Original manifest SHA256: '+hashlib.sha256(original).hexdigest()+'\n'
backup.mkdir()
(backup/'bundle-manifest.sha256').write_bytes(original)
for name,text in changed.items():
    (backup/name).write_bytes((root/name).read_bytes())
    data=text.encode()
    (root/name).write_bytes(data)
    entries[name]=hashlib.sha256(data).hexdigest()
manifest_path.write_bytes(''.join(f'{sha}  {name}\n' for name,sha in entries.items()).encode())
print(json.dumps(dict(changed=list(changed),latest_start='2026-10-26T15:00:00+02:00',
    absolute_finish='2026-10-27T03:00:00+02:00',wall_hours=12,submitted=False),indent=2))
