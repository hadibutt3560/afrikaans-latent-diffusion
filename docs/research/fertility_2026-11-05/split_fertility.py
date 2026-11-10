"""Fertility of the NCHLT splits under the same definition as run_fertility.py: mT5 tokens without
special tokens per whitespace word, pooled over the split. Checks each split file against the SHA-256
recorded in stage4_final_evaluation_2026-10-23/tokenizer_report_copy.json and writes
results/split_fertility.json (never overwrites it). No model is run.
"""
import argparse, hashlib, json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--research', default='C:/Users/SCIENTIST/Documents/GitHub/research')
R = Path(ap.parse_args().research)
TOK = AutoTokenizer.from_pretrained(str(R / 'outputs/longrun-local-inputs/models/mt5-base'))
SPLITS = R / 'data/nchlt_afrikaans/processed/stage_5_7'
REPORT = json.loads((HERE.parent / 'stage4_final_evaluation_2026-10-23/tokenizer_report_copy.json').read_text(encoding='utf-8-sig'))

out = {'definition': 'mT5 tokens without special tokens per whitespace word, pooled over the split'}
for split in ('train', 'valid', 'test'):
    path = SPLITS / f'{split}.jsonl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == REPORT['input'][split]['sha256'], f'{split} does not match the recorded hash'
    tokens = words = passages = 0
    for line in path.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        text = json.loads(line)['text']
        tokens += len(TOK(text, add_special_tokens=False)['input_ids'])
        words += len(text.split())
        passages += 1
    out[split] = dict(passages=passages, tokens=tokens, words=words, fertility=tokens / words, sha256=digest)

with (HERE / 'results/split_fertility.json').open('x', encoding='utf8') as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: (v['fertility'] if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
