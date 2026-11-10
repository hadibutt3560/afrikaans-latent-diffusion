"""Preserve supplied file membership, export exact spans, and audit without cleaning."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
from common import sha, identity, write


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def spans(text, words=48):
    # A line is only an observed delimiter, NEVER asserted to be a document.
    offset = 0
    for line in text.splitlines(keepends=True):
        matches = list(re.finditer(r'\S+', line))
        start = 0
        for k in range(words, len(matches), words):
            end = matches[k].start()
            yield offset+start, offset+end
            start = end
        if matches:
            yield offset+start, offset+len(line)
        offset += len(line)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'audit.json').exists():
        raise FileExistsError('Audit already exists; use a new destination')
    manifest = a.source/'split_manifest.json'; m = json.loads(manifest.read_text(encoding='utf-8-sig'))
    entries = [(s,n) for s,ns in m['splits'].items() for n in ns]
    expected = [n for s,n in entries]; actual = {f.name for f in (a.source/'cleaned').iterdir() if f.is_file()}
    problems = dict(missing=sorted(set(expected)-actual), extra=sorted(actual-set(expected)),
                    repeated_membership=[n for n,c in Counter(expected).items() if c>1])
    ignored_metadata = {n:sha(a.source/'cleaned'/n) for n in problems['extra'] if n == 'Open Notebook.onetoc2'}
    problems['extra'] = [n for n in problems['extra'] if n not in ignored_metadata]
    if any(problems.values()) or len(entries) != m['meta']['n_total']:
        write(a.output/'integrity-failure.json', problems); raise ValueError(problems)
    shutil.copyfile(manifest, a.output/'split_manifest.json')
    db = sqlite3.connect(a.output/'audit-index.sqlite'); db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE spans (id TEXT, split TEXT, file TEXT, exact TEXT, norm TEXT, near TEXT)')
    outputs = {s:(a.output/f'{s}.jsonl').open('w',encoding='utf8',newline='\n') for s in ('train','valid','test')}
    file_rows=[]; counts=Counter(); issues=[]; lines_count=Counter(); totals=Counter()
    for original_split,name in entries:
        split={'val':'valid','validation':'valid'}.get(original_split,original_split)
        if split not in outputs or Path(name).name != name: raise ValueError('Unsafe manifest entry')
        path=a.source/'cleaned'/name; raw=path.read_bytes(); text=raw.decode('utf-8',errors='strict')
        digest=hashlib.sha256(raw).hexdigest(); ls=text.splitlines(); lines_count[len(ls)]+=1
        if not text.strip() or '\ufffd' in text or '\x00' in text or text.startswith('\ufeff'):
            issues.append(dict(file=name,empty=not text.strip(),replacement=text.count('\ufffd'),nul=text.count('\x00'),bom=text.startswith('\ufeff')))
        totals['bytes']+=len(raw); totals['characters']+=len(text); totals['words']+=len(text.split())
        dest=a.output/'raw'/name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(path,dest)
        file_rows.append(dict(file=name,split=split,sha256=digest,bytes=len(raw),characters=len(text),
                              lines=len(ls),blank_lines=sum(not x.strip() for x in ls)))
        for start,end in spans(text):
            passage=text[start:end]; norm=normalized(passage); rid=identity([name,digest,start,end])
            exact=hashlib.sha256(passage.encode()).hexdigest(); words=norm.split()
            # One min-hash over all contiguous 13-word shingles: an indicator only.
            near=min((hashlib.blake2b(' '.join(words[i:i+13]).encode(),digest_size=8).hexdigest()
                      for i in range(len(words)-12)),default=None)
            row=dict(id=rid,split=split,source_file=name,source_sha256=digest,char_start=start,char_end=end,
                     text=passage,text_sha256=exact,observed_unit='line-window',document_id=None)
            outputs[split].write(json.dumps(row,ensure_ascii=False)+'\n'); counts[split]+=1
            db.execute('INSERT INTO spans VALUES (?,?,?,?,?,?)',(rid,split,name,exact,identity(norm),near))
        db.commit()
    for f in outputs.values(): f.close()
    overlap={}
    for key in ('exact','norm','near'):
        db.execute(f'CREATE INDEX idx_{key} ON spans({key})')
        groups=db.execute(f'SELECT {key},COUNT(*),COUNT(DISTINCT split) FROM spans WHERE {key} IS NOT NULL GROUP BY {key} HAVING COUNT(*)>1').fetchall()
        cross=[g for g in groups if g[2]>1]
        examples=[]
        for digest,n,_ in cross[:100]:
            examples.append(dict(hash=digest,count=n,occurrences=db.execute(f'SELECT id,split,file FROM spans WHERE {key}=? LIMIT 20',(digest,)).fetchall()))
        overlap[key]=dict(duplicate_groups=len(groups),duplicate_excess=sum(g[1]-1 for g in groups),
                          cross_split_groups=len(cross),examples=examples)
    file_hashes=defaultdict(list)
    for r in file_rows:file_hashes[r['sha256']].append([r['split'],r['file']])
    duplicate_files=[v for v in file_hashes.values() if len(v)>1]
    report=dict(schema=1,manifest_sha256=sha(manifest),membership_problems=problems,ignored_noncorpus_metadata=ignored_metadata,files=file_rows,
        file_split_counts=dict(Counter(s for s,n in entries)),record_counts=dict(counts),totals=dict(totals),
        utf8_strict=True,encoding_issues=issues,line_count_histogram=dict(lines_count),duplicate_files=duplicate_files,
        overlap=overlap,policy=dict(words_per_window=48,normalization='NONE in exports; NFKC/casefold/whitespace for audit only',
        boundaries='Observed lines only. Original document boundaries are absent/unverifiable; files are concatenated corpus shards.',
        whitespace='Exact Python Unicode character slices; blank-only lines omitted from model rows but retained in raw files.',
        splits='Original file membership unchanged, val renamed valid; no deduplication or filtering.',
        near='MinHash shared 13-word shingle indicator, not exhaustive near-duplicate detection or proof of document independence.'),
        exported_sha256={s:sha(a.output/f'{s}.jsonl') for s in outputs})
    report['requires_dataset_decision']=bool(issues or duplicate_files or overlap['norm']['cross_split_groups'] or overlap['near']['cross_split_groups'])
    write(a.output/'audit.json',report); db.commit(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','overlap')},indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} for k,v in overlap.items()},indent=2))


if __name__=='__main__': main()
