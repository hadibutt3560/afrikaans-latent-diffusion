"""Summarise fixed-checkpoint validation diagnostics; no inference or test access."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import unicodedata

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/ar-validation-audit-20261022'
BENCH=ROOT/'outputs/longrun-local-evaluation'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s]


def norm(text):
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def pooled(rows):
    scores=[r['mgpt'] for r in rows if r['mgpt'] is not None]
    return math.exp(sum(s['nll'] for s in scores)/sum(s['tokens'] for s in scores)) if scores else None


def bootstrap_ppl(rows, seed=20261022):
    # Conditional nonempty interval is explicitly labelled if there are failures.
    scores=[r['mgpt'] for r in rows if r['mgpt'] is not None]
    if not scores:
        return None
    v=np.array([[s['nll'],s['tokens']] for s in scores])
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(v),size=(2000,len(v)))
    sums=v[ix].sum(1)
    values=np.exp(sums[:,0]/sums[:,1])
    return np.quantile(values,[.025,.975]).tolist()


def summary(rows, train):
    words=[r['text'].split() for r in rows]
    empty=sum(not r['text'].strip() for r in rows)
    score_failures=sum(r['mgpt'] is None for r in rows)
    out=dict(samples=len(rows),empty=empty,score_failures=score_failures,
        pooled_ppl=pooled(rows) if not score_failures else None,
        conditional_nonempty_ppl=pooled(rows),
        conditional_sample_ppl_ci95=bootstrap_ppl(rows),
        mean_words=statistics.mean(map(len,words)),median_words=statistics.median(map(len,words)),
        unique_texts=len({r['text'] for r in rows}),exact_training_matches=sum(norm(r['text']) in train for r in rows))
    for n in (1,2,3):
        grams=[tuple(w[i:i+n]) for w in words for i in range(len(w)-n+1)]
        out[f'distinct_{n}']=len(set(grams))/len(grams) if grams else None
    rep=[]
    for w in words:
        grams=[tuple(w[i:i+3]) for i in range(len(w)-2)]
        rep.append(1-len(set(grams))/len(grams) if grams else 0.)
    out['mean_repeated_trigram_fraction']=statistics.mean(rep)
    for key in ('eos','length_cap','sentinel'):
        out[key+'_fraction']=statistics.mean(r[key] for r in rows) if all(key in r for r in rows) else None
    bands={}
    for label,lo,hi in [('1-10',1,10),('11-20',11,20),('21-35',21,35),('36+',36,10000)]:
        selected=[r for r in rows if lo<=len(r['text'].split())<=hi]
        bands[label]=dict(samples=len(selected),ppl=pooled(selected))
    out['word_length_bands']=bands
    return out


def main():
    if (OUT/'analysis.json').exists():
        raise FileExistsError('Analysis already exists; preserve it')
    train={norm(r['text']) for r in jsonl(ROOT/'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl')}
    groups={name:jsonl(OUT/f'{name}-scored.jsonl') for name in ('ar-cap127','ar-cap64','ar-historical100')}
    full_benchmark={}
    for model in ('small','published'):
        for seed in range(42,47):
            name=f'{model}-seed{seed}'
            record=read(BENCH/'runs'/name/'validation-mgpt.json')
            assert record['revision']=='40897bd7c8b47a76802c411108ca6220438b8b40'
            groups[name]=record['rows'][:256]
            assert [r['seed'] for r in groups[name]]==list(range(30000,30256))
            full_benchmark[name]=dict(count=record['count'],pooled_ppl=record['token_weighted_perplexity'])
    groups['real-validation']=read(BENCH/'validation-reference-mgpt.json')['rows'][:256]
    summaries={name:summary(rows,train) for name,rows in groups.items()}
    val=read(OUT/'full-validation.json')
    docs=defaultdict(lambda:np.zeros(2))
    for row in val['rows']:
        docs[row['document_id']]+=np.array([row['nll'],row['tokens']])
    values=np.array(list(docs.values()))
    rng=np.random.default_rng(20261022)
    indices=rng.integers(0,len(values),size=(2000,len(values)))
    totals=values[indices].sum(1)
    ces=totals[:,0]/totals[:,1]
    val_summary={k:v for k,v in val.items() if k!='rows'}
    val_summary.update(documents=len(values),document_bootstrap_ce_ci95=np.quantile(ces,[.025,.975]).tolist(),
        document_bootstrap_ppl_ci95=np.quantile(np.exp(ces),[.025,.975]).tolist())
    a,b=groups['ar-cap127'],groups['ar-cap64']
    assert len(a)==len(b)==256 and [r['seed'] for r in a]==[r['seed'] for r in b]
    prefix=sum(x['token_ids'][:len(y['token_ids'])]==y['token_ids'] for x,y in zip(a,b))
    paired=dict(seed_pairs=256,identical_prefix_pairs=prefix,full_identical_text_pairs=sum(x['text']==y['text'] for x,y in zip(a,b)))
    if all(r['mgpt'] is not None for r in a+b):
        v=np.array([[x['mgpt']['nll'],x['mgpt']['tokens'],y['mgpt']['nll'],y['mgpt']['tokens']] for x,y in zip(a,b)])
        ix=np.random.default_rng(20261022).integers(0,256,size=(2000,256))
        sums=v[ix].sum(1)
        diffs=np.exp(sums[:,0]/sums[:,1])-np.exp(sums[:,2]/sums[:,3])
        paired.update(ppl_difference_cap127_minus_cap64=pooled(a)-pooled(b),
                      paired_sample_bootstrap_ci95=np.quantile(diffs,[.025,.975]).tolist())
    seed_summaries={}
    for model in ('small','published'):
        vals=[summaries[f'{model}-seed{s}']['pooled_ppl'] for s in range(42,47)]
        seed_summaries[model]=dict(training_seeds=5,mean_ppl=statistics.mean(vals),sample_sd_ppl=statistics.stdev(vals),
            descriptive_mean_ppl_to_AR_ratio=statistics.mean(vals)/summaries['ar-cap127']['pooled_ppl'] if summaries['ar-cap127']['pooled_ppl'] else None,
            inference='Contextual fixed-checkpoint comparison only; not a matched hypothesis test')
    history=jsonl(ROOT/'outputs/cluster-results/ar-baseline-57759/training_metrics.jsonl')
    checks=[r for r in history if 'validation_cross_entropy' in r]
    trajectory=dict(logged_steps=len(history),validation_checks=len(checks),first=checks[0],
        best=min(checks,key=lambda r:r['validation_cross_entropy']),last=checks[-1],
        interpretation='Repeated selection on 128 validation examples; later deterioration is descriptive and does not justify test-based reselection.')
    report=dict(groups=summaries,diffusion_training_seed_summaries=seed_summaries,full_diffusion_benchmark_context=full_benchmark,
        training_trajectory=trajectory,
        full_validation=val_summary,paired_cap_diagnostic=paired,
        bootstrap_replicates=2000,bootstrap_seed=20261022,interval_type='percentile, conditional on fixed weights; validation CE resamples source documents',
        hypothesis_assessed=False,human_evaluation_completed=False,test_read=False)
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    fields=['group','samples','pooled_ppl','mean_words','distinct_2','mean_repeated_trigram_fraction','exact_training_matches','eos_fraction','length_cap_fraction','empty']
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for name,s in summaries.items():
            writer.writerow(dict(group=name,**{k:s[k] for k in fields if k!='group'}))
    chosen=random.Random(20261022).sample(range(256),20)
    lines=['# Fixed sample for AI-assisted diagnostic inspection','',
        'Selection: random.Random(20261022).sample(range(256),20). No human ratings have been collected.','']
    for i in chosen:
        row=a[i]
        lines.extend([f"## Seed {row['seed']}", '', row['text'], '',
                      f"mGPT PPL: {row['mgpt']['perplexity'] if row['mgpt'] else 'failed'}. EOS: {row['eos']}. Cap: {row['length_cap']}.",''])
    lines+=['# Targeted examples (not prevalence estimates)','']
    extremes=sorted([r for r in a if r['mgpt']],key=lambda r:r['mgpt']['perplexity'],reverse=True)[:5]
    for row in extremes:
        lines.extend([f"## High scorer perplexity: seed {row['seed']}",'',row['text'],''])
    copied=[r for r in a if norm(r['text']) in train]
    for row in copied[:5]:
        lines.extend([f"## Exact training match: seed {row['seed']}",'',row['text'],''])
    (HERE/'SAMPLES_FOR_INSPECTION.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    # A small unscored review sheet; no claim to implement the final human protocol.
    review=[dict(group='ar-cap127',**a[i]) for i in chosen]
    for model in ('small','published'):
        for seed in range(42,47):
            inds=random.Random(20261022+seed).sample(range(256),4)
            review.extend(dict(group=f'{model}-seed{seed}',**groups[f'{model}-seed{seed}'][i]) for i in inds)
    random.Random(20261023).shuffle(review)
    with (OUT/'unrated-blind-review.csv').open('w',newline='',encoding='utf8') as f, (OUT/'review-key.csv').open('w',newline='',encoding='utf8') as key:
        w=csv.writer(f); k=csv.writer(key)
        w.writerow(['id','text','grammar','coherence','notes']); k.writerow(['id','group','generation_seed'])
        for i,row in enumerate(review,1):
            w.writerow([f'V{i:03d}',row['text'],'','','']); k.writerow([f'V{i:03d}',row['group'],row['seed']])
    (OUT/'analysis-source-sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','full_diffusion_benchmark_context')},indent=2))
    print(json.dumps({k:{f:v[f] for f in ('pooled_ppl','mean_words','distinct_2','exact_training_matches','length_cap_fraction','empty')} for k,v in summaries.items()},indent=2))


if __name__=='__main__':
    main()
