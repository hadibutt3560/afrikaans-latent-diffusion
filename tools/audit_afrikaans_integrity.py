"""Read-only split and saved-generation audit; never load test text for tuning."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    splits={}
    report={'splits':{},'overlap':{},'generation':{}}
    for name in ('train','valid','test'):
        path=Path(f'data/nchlt_afrikaans/processed/stage_5_7/{name}.jsonl')
        rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        texts=[normalized(r['text']) for r in rows]
        splits[name]=(set(texts),{r['document_id'] for r in rows})
        report['splits'][name]={'records':len(rows),'documents':len(splits[name][1]),
            'normalized_duplicates':len(texts)-len(set(texts)),
            'empty':sum(not t for t in texts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for a,b in (('train','valid'),('train','test'),('valid','test')):
        report['overlap'][f'{a}-{b}']={'exact_text':len(splits[a][0]&splits[b][0]),
            'document_ids':len(splits[a][1]&splits[b][1])}
    root=Path('outputs/nchlt-afrikaans/self-condition-recovered-20260930')
    for name in ('control','aligned'):
        path=root/name/'trajectories.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        final={}
        for row in rows:
            if row['seed'] not in final or row['step']>final[row['seed']]['step']:
                final[row['seed']]=row
        chosen=list(final.values())
        words=[re.findall(r'\w+|[^\w\s]',r['text'],flags=re.UNICODE) for r in chosen]
        pairs=Counter(tuple(w[i:i+2]) for w in words for i in range(len(w)-1))
        repeat=[]
        for w in words:
            grams=[tuple(w[i:i+3]) for i in range(len(w)-2)]
            repeat.append(0 if not grams else 1-len(set(grams))/len(grams))
        report['generation'][name]={'samples':len(chosen),'final_step':sorted({r['step'] for r in chosen}),
            'unique_texts':len({r['text'] for r in chosen}),
            'mean_decoder_tokens':sum(r['decoder_tokens'] for r in chosen)/len(chosen),
            'mean_regex_tokens':sum(map(len,words))/len(words),
            'distinct_2':len(pairs)/sum(pairs.values()),
            'mean_within_output_repeated_trigram_fraction':sum(repeat)/len(repeat),
            'length_cap':sum(r['reached_length_cap'] for r in chosen),
            'eos':sum(r['ended_with_eos'] for r in chosen),
            'exact_training_matches':sum(normalized(r['text']) in splits['train'][0] for r in chosen),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report['limits']=['Test text examined for split integrity only; no test quality evaluation.',
        'Exact matching does not rule out near-duplicates or partial memorization.',
        'Diversity is corpus-level Unicode regex distinct-2; repetition is within-output trigram reuse.',
        'Paired variants are not independent training replicates. EOS is not a grammar judgment.']
    (args.output_dir/'integrity.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
