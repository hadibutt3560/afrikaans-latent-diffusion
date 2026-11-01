"""Read-only audit and paired bootstrap of downloaded decoder-aware results."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys


def weighted(rows):
    return math.exp(sum(r['mgpt']['nll'] for r in rows)/sum(r['mgpt']['tokens'] for r in rows))


def main():
    root=Path(sys.argv[1])
    out=Path(sys.argv[2])
    if out.exists():
        raise FileExistsError(out)
    protocol=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert summary['status']=='completed_matched'
    assert summary['pinned_external_scoring']=='completed'
    groups={}
    hashes={}
    for arm in ('frozen','control','decoder_ce'):
        path=root/f'{arm}-scored.json'
        rows=json.loads(path.read_text(encoding='utf8'))
        source=json.loads((root/f'{arm}-samples.json').read_text(encoding='utf8'))
        assert [(r['seed'],r['text']) for r in rows]==[(r['seed'],r['text']) for r in source]
        assert [r['seed'] for r in rows]==protocol['sampling_seeds']
        assert all(math.isfinite(r['mgpt']['nll']) and r['mgpt']['tokens']>0 for r in rows)
        groups[arm]=rows
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in ('control','decoder_ce'):
        rows=[json.loads(line) for line in (root/arm/'training.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,601))
        assert all(math.isfinite(r[k]) for r in rows for k in ('velocity_loss','decoder_ce','grad_norm'))
        assert summary['training'][arm]['step']==600
    reports={}
    for candidate,control in (('decoder_ce','control'),('control','frozen'),('decoder_ce','frozen')):
        a,b=groups[candidate],groups[control]
        rng=random.Random(20261001)
        ratios=[]
        for _ in range(10000):
            ids=rng.choices(range(len(a)),k=len(a))
            ratios.append(weighted([a[i] for i in ids])/weighted([b[i] for i in ids]))
        ratios.sort()
        reports[f'{candidate}_vs_{control}']=dict(ratio=weighted(a)/weighted(b),
            paired_bootstrap_95_interval=[ratios[249],ratios[9749]],
            lower_ppl_pairs=sum(x['mgpt']['perplexity']<y['mgpt']['perplexity'] for x,y in zip(a,b)),
            identical_outputs=sum(x['text']==y['text'] for x,y in zip(a,b)))
    out.mkdir(parents=True)
    report=dict(status='passed',scored_samples_per_arm=30,matched_updates=600,
        source_hashes=hashes,comparisons=reports,
        limits=['Single training seed; paired generation-seed bootstrap, not training replication.',
            'Exploratory comparisons not adjusted for multiple comparisons.',
            'External perplexity is not a human coherence score. Human review pending.'])
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
