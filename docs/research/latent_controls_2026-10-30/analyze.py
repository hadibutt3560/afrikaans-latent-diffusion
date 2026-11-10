"""Analysis for the latent-prior controls (PROTOCOL.md sections 5-7). CPU only; no model is run.

Usage: python analyze.py <outputs/latent-controls-YYYYMMDD>
Writes analysis.json and ALL_OUTPUTS.md into that directory and never overwrites them.
"""
import sys, json, math, hashlib, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TRAIN=ROOT/'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl'
REFERENCE=ROOT/'outputs/longrun-local-evaluation/validation-reference-mgpt.json'
RUNS=ROOT/'outputs/longrun-local-evaluation/runs'
PROTOCOL_SHA='14319d16a373e1d358ace2f779f4f1181260bbb3f3a6cd0033978c184c088994'
BANDS=[(1,10),(11,20),(21,35),(36,10**9)]   # the 27 October audit bands
WEIGHTS=np.array([.140,.223,.349,.288])     # band shares of all 1,000 human-reference rows (section 5)
MIN_BAND=20
T_PRIMARY,T_EXPLORATORY=3.4954,2.7764       # Student t, 4 df, 0.9875 and 0.975
REPLICATES=10000
SEED_CONTROL,SEED_DOCUMENT=20261030,20261031
SIZES={'small':[f'small-seed{s}' for s in range(42,47)],'large':[f'published-seed{s}' for s in range(42,47)]}
CONTROLS=('C1','C2','C3')
sys.stdout.reconfigure(encoding='utf8')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def rows_of(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]
def norm(text): return ' '.join(unicodedata.normalize('NFC',text).casefold().split())
def band(words): return next(i for i,(a,b) in enumerate(BANDS) if a<=words<=b)
def scored(rows): return [r for r in rows if r.get('mgpt')]

def band_arrays(rows):
    out=[]
    for k in range(len(BANDS)):
        rs=[r for r in scored(rows) if band(len(r['text'].split()))==k]
        out.append((np.array([r['mgpt']['nll'] for r in rs]),np.array([r['mgpt']['tokens'] for r in rs])))
    return out

def support(counts):
    if min(counts)==0: return 'not_estimable'
    return 'ok' if min(counts)>=MIN_BAND else 'insufficient_support'

def standardised(arrays):
    return math.exp(float(sum(w*a.sum()/t.sum() for w,(a,t) in zip(WEIGHTS,arrays))))

def bank_stats(rows,train_texts,train_grams):
    s=scored(rows)
    counts=[sum(band(len(r['text'].split()))==k for r in s) for k in range(len(BANDS))]
    status=support(counts)
    words=[r['text'].split() for r in rows]
    bigrams=[tuple(w[i:i+2]) for w in words for i in range(len(w)-1)]
    unigrams=Counter(t for r in rows for t in norm(r['text']).split())
    total=sum(unigrams.values())
    grams=[g for r in rows for t in [norm(r['text']).split()] for g in (' '.join(t[i:i+8]) for i in range(len(t)-7))]
    per_token=np.array([r['mgpt']['nll']/r['mgpt']['tokens'] for r in s])
    out=dict(n=len(rows),scored=len(s),empty=sum(not r.get('mgpt') for r in rows),
        raw_ppl=math.exp(sum(r['mgpt']['nll'] for r in s)/sum(r['mgpt']['tokens'] for r in s)),
        band_counts=counts,band_support=status,standardised_ppl=standardised(band_arrays(rows)) if status=='ok' else None,
        distinct2=len(set(bigrams))/max(1,len(bigrams)),
        unigram_entropy_bits=float(-sum(c/total*math.log2(c/total) for c in unigrams.values())),
        training_copies=sum(norm(r['text']) in train_texts for r in rows),
        eightgram_overlap=sum(g in train_grams for g in grams)/len(grams) if grams else None,eightgram_total=len(grams),
        rows_under_8_words=sum(len(w)<8 for w in words),mean_words=float(np.mean([len(w) for w in words])),
        nll_per_token_quantiles=dict(zip(('p05','p25','p50','p75','p95'),np.percentile(per_token,[5,25,50,75,95]).tolist())))
    for flag in ('eos','length_cap','sentinel'):
        if all(flag in r for r in rows): out[flag+'_count']=sum(bool(r[flag]) for r in rows)
    return out

def control_bootstrap(rows):
    """Resample within each length band, keeping observed band counts and the fixed weights (section 6)."""
    rng=np.random.default_rng(SEED_CONTROL)
    log_ppl=np.zeros(REPLICATES)
    for w,(nll,tok) in zip(WEIGHTS,band_arrays(rows)):
        idx=rng.integers(0,len(nll),size=(REPLICATES,len(nll)))
        log_ppl+=w*nll[idx].sum(1)/tok[idx].sum(1)
    return np.exp(log_ppl)

def contrast(runs,control,t,label):
    bad=[k for k,v in list(runs.items())+[(label,control)] if v['band_support']!='ok']
    if bad:
        return dict(classification='insufficient_support',banks=[dict(bank=k,band_counts=(runs.get(k) or control)['band_counts'],
                    support=(runs.get(k) or control)['band_support']) for k in bad])
    values=np.array([v['standardised_ppl'] for v in runs.values()])
    variance=float(control['bootstrap_variance'])
    d=float(values.mean()-control['standardised_ppl'])
    half=t*math.sqrt(values.var(ddof=1)/len(values)+variance)
    low,high=d-half,d+half
    cls='lower' if high<0 else 'higher' if low>0 else 'not_established'
    return dict(classification=cls,difference=d,low=low,high=high,t=t,run_values=values.tolist(),run_sd=float(values.std(ddof=1)),
                control_value=control['standardised_ppl'],control_bootstrap_variance=variance,
                interval='approximate t-based; between-run variance plus estimated control sampling variance')

def document_bootstrap(h,r):
    """H and R: documents resampled with replacement, all their rows kept, R paired with H (section 6)."""
    docs=sorted({x['document_id'] for x in h}); index={d:i for i,d in enumerate(docs)}
    def tables(rows):
        nll=np.zeros((len(docs),len(BANDS))); tok=np.zeros_like(nll); cnt=np.zeros_like(nll)
        for x in scored(rows):
            i,k=index[x['document_id']],band(len(x['text'].split()))
            nll[i,k]+=x['mgpt']['nll']; tok[i,k]+=x['mgpt']['tokens']; cnt[i,k]+=1
        return nll,tok,cnt
    rng=np.random.default_rng(SEED_DOCUMENT)
    draws=rng.integers(0,len(docs),size=(REPLICATES,len(docs)))
    weights=np.stack([np.bincount(d,minlength=len(docs)) for d in draws]).astype(float)
    out={}
    for name,rows in (('H',h),('R',r)):
        nll,tok,cnt=tables(rows)
        sn,st,sc=weights@nll,weights@tok,weights@cnt
        raw=np.exp(sn.sum(1)/st.sum(1))
        empty=(sc==0).any(1)
        result=dict(raw_ppl_ci95=np.percentile(raw,[2.5,97.5]).tolist(),replicates_with_empty_band=int(empty.sum()),
                    fraction_with_empty_band=float(empty.mean()))
        if empty.any():
            result['standardised_ppl_ci95']=None
            result['standardised_note']='withheld: at least one replicate has an empty band'
        else:
            result['standardised_ppl_ci95']=np.percentile(np.exp((WEIGHTS*sn/st).sum(1)),[2.5,97.5]).tolist()
        out[name]=result
    return out

def reconstruction_summary(r):
    m=[x['model_metrics'] for x in r]
    expected=sum(x['number_expected'] for x in m)
    predicted=sum(x['number_predicted'] for x in m)
    matched=sum(x['number_matched'] for x in m)
    return dict(n=len(r),exact_model_view=sum(x['exact'] for x in m),exact_rate=sum(x['exact'] for x in m)/len(m),
                mean_chrf=float(np.mean([x['chrf'] for x in m])),numeral_recall=matched/expected if expected else None,
                numeral_precision=matched/predicted if predicted else None,eos_count=sum(x['eos'] for x in r),cap_count=sum(x['length_cap'] for x in r))

def examples(rows):
    n=len(rows)
    return [dict(index=i,text=rows[i]['text'],**({'source':rows[i]['source']} if 'source' in rows[i] else {})) for i in (0,n//4,n//2,3*n//4)]

def main():
    out=Path(sys.argv[1]).resolve()
    contract=read(out/'contract.json')
    assert contract['protocol_sha256_lf']==PROTOCOL_SHA
    assert (out/'runtime.json').exists(), 'Inference did not complete'
    for name,h in contract['bank_sha256'].items(): assert digest(RUNS/name/'validation-mgpt.json')==h, name
    assert digest(REFERENCE)==contract['reference_sha256']
    manifest=read(out/'manifest.json')
    train=[r['text'] for r in rows_of(TRAIN)]
    train_texts={norm(t) for t in train}
    train_grams={' '.join(t[i:i+8]) for x in train for t in [norm(x).split()] for i in range(len(t)-7)}
    reference={r['id']:r for r in read(REFERENCE)['rows']}
    h=[reference[x['id']] for x in manifest['H']]
    banks={name:rows_of(out/f'scored-{name}.jsonl') for name in CONTROLS+('R',)}
    assert all(len(banks[c])==1000 for c in CONTROLS) and [x['id'] for x in banks['R']]==[x['id'] for x in h]
    banks['H']=h
    runs={name:read(RUNS/name/'validation-mgpt.json')['rows'] for names in SIZES.values() for name in names}
    stats={name:bank_stats(rows,train_texts,train_grams) for name,rows in {**banks,**runs}.items()}
    for c in CONTROLS:
        if stats[c]['band_support']=='ok':
            b=control_bootstrap(banks[c])
            stats[c]['bootstrap_variance']=float(b.var(ddof=1))
            stats[c]['bootstrap_ci95']=np.percentile(b,[2.5,97.5]).tolist()
    summary={}
    for size,names in SIZES.items():
        summary[size]={key:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))) for key in
            ('raw_ppl','distinct2','unigram_entropy_bits','training_copies','eightgram_overlap','mean_words')
            for v in [[stats[n][key] for n in names]]}
        values=[stats[n]['standardised_ppl'] for n in names]
        summary[size]['standardised_ppl']=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1))) if None not in values else None
    primary={size:contrast({n:stats[n] for n in names},stats['C3'],T_PRIMARY,'C3') for size,names in SIZES.items()}
    exploratory={f'{size}-{c}':contrast({n:stats[n] for n in names},stats[c],T_EXPLORATORY,c) for size,names in SIZES.items() for c in ('C1','C2')}
    result=dict(protocol_sha256_lf=PROTOCOL_SHA,analysis_script_sha256=digest(__file__),weights=WEIGHTS.tolist(),minimum_band_rows=MIN_BAND,
        primary=primary,primary_note='Two contrasts at 97.5%; Bonferroni nominal family-wise level 95%; coverage approximate with five runs.',
        exploratory=exploratory,exploratory_note='95% intervals; exploratory only.',diffusion_summary=summary,banks=stats,
        human_and_reconstruction_bootstrap=document_bootstrap(h,banks['R']),reconstruction=reconstruction_summary(banks['R']),
        diversity_note='H and R have fewer rows than the 1,000-row banks; their distinct-2 and entropy are descriptive and sample-size-sensitive.',
        c2_record=read(out/'c2-record.json'),
        examples={**{c:examples(banks[c]) for c in CONTROLS+('R',)},**{n:examples(runs[n]) for n in ('small-seed42','published-seed42')}},
        examples_note='Fixed-rule rows 0, n/4, n/2, 3n/4; AI-assisted reading, not human evaluation.')
    with (out/'analysis.json').open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# All decoded outputs','','Every text decoded in this run. Diffusion outputs are in the existing validation banks and are not repeated here.','']
    for name in CONTROLS+('R',):
        lines+=[f'## {name}','']
        for i,x in enumerate(banks[name]):
            lines.append(f'{i}. ' + (f"**Source:** {x['source']}  \n   **Output:** " if name=='R' else '') + (x['text'] or '*(empty)*'))
        lines.append('')
    with (out/'ALL_OUTPUTS.md').open('x',encoding='utf8') as f: f.write('\n'.join(lines))
    print(json.dumps(dict(primary={k:v['classification'] for k,v in primary.items()}),indent=2))

if __name__=='__main__': main()
