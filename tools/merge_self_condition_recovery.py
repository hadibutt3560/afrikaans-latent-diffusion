"""Assemble completed seed records in a new folder, preserving original runs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

original, remaining, output = map(Path, sys.argv[1:4])
if output.exists():
    raise FileExistsError(output)
sources = [original/'aligned', remaining]
metadata = [json.loads((s/'metadata.json').read_text()) for s in sources]
for key in ('diffusion_sha256','autoencoder_sha256','sampler','inference_steps','decoding',
            'normalize_sampling_self_condition','precision','gpu','observe_every'):
    if metadata[0][key] != metadata[1][key]:
        raise ValueError(f'Mismatch: {key}')
if metadata[1]['status'] != 'completed':
    raise ValueError('Remaining run incomplete')
rows = []
provenance = []
for source in sources:
    path = source/'trajectories.jsonl'
    subset = [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]
    rows.extend(subset)
    provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
expected = {(s,t) for s in range(112,142) for t in range(1,51)}
if len(rows) != 1500 or {(r['seed'],r['step']) for r in rows} != expected:
    raise ValueError('Missing or duplicate recovered records')
rows.sort(key=lambda r:(r['seed'],r['step']))
output.mkdir()
shutil.copytree(original/'control',output/'control')
shutil.copytree(original/'quality-control',output/'quality-control')
aligned = output/'aligned'
aligned.mkdir()
(aligned/'trajectories.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
meta = dict(metadata[0],status='completed',seeds=list(range(112,142)),recovery_sources=provenance)
meta.pop('elapsed_sampling_and_decoding_seconds',None)
meta.pop('seed_summaries',None)
(aligned/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
print(f'Validated and assembled 1500 aligned records: {output}')
