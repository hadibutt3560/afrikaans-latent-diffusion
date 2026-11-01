"""Turn the downloaded Gov-ZA and Vuk'uzenzele Afrikaans documents into NCHLT-style passages.

Steps, in order (recorded in the manifest):
1. Layout repair. Gov-ZA lost its line breaks, so numbered headings and sentences run together
   ("omgewing1.1. Die", "verkiesing.Die"). A newline is inserted at those joins only.
2. NCHLT stage 5 normalisation and contact masking (tools/finalize_nchlt_afrikaans.py, unchanged).
3. NCHLT stage 6 segmentation: each line split to at most 50 whitespace words at sentence
   punctuation where possible; segments under three words removed.
4. Exact duplicates removed, within the new text and against every NCHLT split.
5. Leakage filter: any document with a segment sharing a 12-word sequence with NCHLT validation
   or test is removed whole.
6. 5% of the remaining documents (seeded) are held out as a monitoring slice; they are never
   used for training or checkpoint selection.

Output: data/extra_afrikaans/processed/ (extra_train.jsonl, extra_monitor.jsonl, audit files,
manifest.json). Refuses to overwrite.
"""
import hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from finalize_nchlt_afrikaans import comparison_key, normalize_and_mask_text, split_long_line

RAW = ROOT / 'data/extra_afrikaans/raw'
OUT = ROOT / 'data/extra_afrikaans/processed'
NCHLT = ROOT / 'data/nchlt_afrikaans/processed/stage_5_7'
MAX_WORDS, MIN_WORDS, NGRAM = 50, 3, 12
MONITOR_FRACTION, SEED = 0.05, 20261027
# A numbered heading glued to the previous word ("omgewing1.1. Die") or a sentence glued to the
# next one ("verkiesing.Die"). Only these joins are touched.
HEADING_JOIN = re.compile(r'(?<=[^\s\d.])(?=\d{1,2}(?:\.\d{1,2})+\.?\s?[A-ZÀ-ÖØ-Þ])')
SENTENCE_JOIN = re.compile(r'(?<=[a-zà-öø-ÿ][.!?])(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def grams(text):
    words = comparison_key(text).split()
    return {' '.join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def repair_layout(text):
    repaired, headings = HEADING_JOIN.subn('\n', text)
    repaired, sentences = SENTENCE_JOIN.subn('\n', repaired)
    return repaired, headings, sentences


def segment(document):
    repaired, headings, sentences = repair_layout(document['text'])
    normalized, audit = normalize_and_mask_text(repaired)
    audit.update(heading_joins_split=headings, sentence_joins_split=sentences)
    segments, short = [], 0
    for line_index, line in enumerate(normalized.split('\n')):
        for chunk_index, chunk in enumerate(split_long_line(line, MAX_WORDS)):
            if len(chunk.split()) < MIN_WORDS:
                short += 1
                continue
            segments.append(dict(line_index=line_index, chunk_index=chunk_index, text=chunk))
    audit['short_segments_removed'] = short
    return segments, audit


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    raw_manifest = json.loads((RAW / 'manifest.json').read_text(encoding='utf8'))
    nchlt = {split: read_jsonl(NCHLT / f'{split}.jsonl') for split in ('train', 'valid', 'test')}
    nchlt_keys = {comparison_key(r['text']) for rows in nchlt.values() for r in rows}
    heldout_grams = set()
    for split in ('valid', 'test'):
        for r in nchlt[split]:
            heldout_grams |= grams(r['text'])

    documents, audits = [], Counter()
    for source in ('gov-za', 'vukuzenzele'):
        for d in read_jsonl(RAW / source / 'documents.jsonl'):
            document_id = 'extra-afr-' + hashlib.sha256(f"{source}:{d['source_index']}".encode()).hexdigest()[:40]
            segments, audit = segment(d)
            audits.update({f'{source}/{k}': v for k, v in audit.items()})
            documents.append(dict(document_id=document_id, source=source, source_index=d['source_index'], segments=segments))

    seen, removed_duplicates, leaked_documents = set(), [], []
    kept = []
    for d in documents:
        if any(grams(s['text']) & heldout_grams for s in d['segments']):
            leaked_documents.append(dict(document_id=d['document_id'], source=d['source'], source_index=d['source_index']))
            continue
        rows = []
        for s in d['segments']:
            key = comparison_key(s['text'])
            if key in nchlt_keys or key in seen:
                removed_duplicates.append(dict(document_id=d['document_id'], text=s['text'],
                                               reason='nchlt_duplicate' if key in nchlt_keys else 'extra_duplicate'))
                continue
            seen.add(key)
            segment_id = 'extra-afr-seg-' + hashlib.sha256(
                f"{d['document_id']}:{s['line_index']}:{s['chunk_index']}".encode()).hexdigest()[:40]
            rows.append(dict(id=segment_id, document_id=d['document_id'], source=d['source'], language='afr',
                             line_index=s['line_index'], segment_index=s['chunk_index'], text=s['text'],
                             word_count=len(s['text'].split())))
        if rows:
            kept.append((d, rows))

    documents_ids = sorted(d['document_id'] for d, _ in kept)
    random.Random(SEED).shuffle(documents_ids)
    monitor_ids = set(documents_ids[:max(1, round(MONITOR_FRACTION * len(documents_ids)))])
    train = [r for d, rows in kept if d['document_id'] not in monitor_ids for r in rows]
    monitor = [r for d, rows in kept if d['document_id'] in monitor_ids for r in rows]

    def write(name, rows):
        with (OUT / name).open('x', encoding='utf8') as f:
            for r in rows:
                f.write(json.dumps(dict(r, split='extra_train' if name.startswith('extra_train') else r.get('split', 'audit')),
                                   ensure_ascii=False, sort_keys=True) + '\n')

    write('extra_train.jsonl', train)
    write('extra_monitor.jsonl', monitor)
    write('removed_duplicates.jsonl', removed_duplicates)
    write('leaked_documents.jsonl', leaked_documents)

    def summary(rows):
        return dict(passages=len(rows), documents=len({r['document_id'] for r in rows}), words=sum(r['word_count'] for r in rows),
                    by_source=dict(Counter(r['source'] for r in rows)))

    manifest = dict(
        raw_manifest_sha256=digest(RAW / 'manifest.json'), raw_sources=raw_manifest,
        nchlt_sha256={s: digest(NCHLT / f'{s}.jsonl') for s in ('train', 'valid', 'test')},
        policy=dict(max_words=MAX_WORDS, min_words=MIN_WORDS, leakage_ngram=NGRAM, monitor_fraction=MONITOR_FRACTION, seed=SEED,
                    normalisation='tools/finalize_nchlt_afrikaans.py normalize_and_mask_text and split_long_line, unchanged'),
        preprocessing_code_sha256={'prepare_extra.py': digest(__file__), 'finalize_nchlt_afrikaans.py': digest(ROOT / 'tools/finalize_nchlt_afrikaans.py')},
        audits=dict(audits), leaked_documents_removed=len(leaked_documents), duplicates_removed=dict(Counter(r['reason'] for r in removed_duplicates)),
        extra_train=summary(train), extra_monitor=summary(monitor),
        files={n: digest(OUT / n) for n in ('extra_train.jsonl', 'extra_monitor.jsonl', 'removed_duplicates.jsonl', 'leaked_documents.jsonl')})
    with (OUT / 'manifest.json').open('x', encoding='utf8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ('leaked_documents_removed', 'duplicates_removed', 'extra_train', 'extra_monitor')}, indent=2))


if __name__ == '__main__':
    main()
