"""Freeze the three NCHLT validation panels used by the autoencoder screen (PROTOCOL.md section 3).

The 251 validation documents are already partitioned by the seed-91427 shuffle used since the
22 October audit:
  positions 0-63    dev documents      -> selection panel: all their passages (checkpoint choice, arm choice)
  positions 64-127  reserve documents  -> reserve panel: the 103 passages fixed on 22 October (winner and B0, once)
  positions 128-250 held-out documents -> V panel: the 194 passages of the 27 October diagnostic (all arms)
No document appears in two panels. Writes data/ae_screen_panels/. Refuses to overwrite.
"""
import hashlib, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VALID = ROOT / 'data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl'
STAGE1 = ROOT / 'outputs/reconstruction-audit-20261022/manifest.json'
DIAGNOSTIC = ROOT / 'outputs/train-heldout-recon-20261027/manifest.json'
OUT = ROOT / 'data/ae_screen_panels'
KEYS = ('id', 'document_id', 'text')


def main():
    rows = [json.loads(line) for line in VALID.read_text(encoding='utf8').splitlines() if line.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[r['document_id']].append(r)
    docs = sorted(groups)
    random.Random(91427).shuffle(docs)
    stage1 = json.loads(STAGE1.read_text(encoding='utf8'))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding='utf8'))
    assert {r['document_id'] for r in stage1['selected']} == set(docs[:64])
    assert {r['document_id'] for r in stage1['reserved']} <= set(docs[64:128])
    assert {r['document_id'] for r in diagnostic['V']} <= set(docs[128:])
    panels = dict(
        selection=[{k: r[k] for k in KEYS} for d in docs[:64] for r in groups[d]],
        reserve=[{**{k: r[k] for k in KEYS}, 'bin': r['bin']} for r in stage1['reserved']],
        V=[{**{k: r[k] for k in KEYS}, 'bin': r['bin']} for r in diagnostic['V']],
    )
    docsets = {k: {r['document_id'] for r in v} for k, v in panels.items()}
    assert not (docsets['selection'] & docsets['reserve']) and not (docsets['selection'] & docsets['V']) and not (docsets['reserve'] & docsets['V'])
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, panel in panels.items():
        path = OUT / f'{name}.jsonl'
        with path.open('x', encoding='utf8') as f:
            for r in panel:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        manifest[name] = dict(passages=len(panel), documents=len(docsets[name]), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
