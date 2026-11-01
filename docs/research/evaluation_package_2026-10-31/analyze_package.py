"""Statistics for the evaluation package (PROTOCOL.md). Helpers are imported by run_package.py; run directly for the final analysis."""
import json, math, random, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/evaluation-package-20261101'
PHASE_A=ROOT/'outputs/latent-controls-20261030'
BANDS=[(1,10),(11,20),(21,35),(36,10**9)]   # Phase A bands, unchanged
WEIGHTS=np.array([.140,.223,.349,.288])     # Phase A reference weights, unchanged
MIN_BAND=20
T4_9875, T4_975 = 3.4954, 2.7764
SIZES={'small':[f'small-seed{s}' for s in range(42,47)],'large':[f'published-seed{s}' for s in range(42,47)]}
RUNS=SIZES['small']+SIZES['large']
CRITERION=1.20

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def rows_of(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]
def norm(text): return ' '.join(unicodedata.normalize('NFC',text).casefold().split())   # compare_generation.py norm()
def band(words): return next(i for i,(a,b) in enumerate(BANDS) if a<=words<=b)
def nwords(r): return len(r['text'].split())
def load_train(): return rows_of(ROOT/'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl')

# ---------------------------------------------------------------- perplexity (section 7)
def scored(rows,scorer): return [r for r in rows if r.get(scorer)]

def band_arrays(rows,scorer):
    out=[]
    for k in range(len(BANDS)):
        rs=[r for r in scored(rows,scorer) if nwords(r)>0 and band(nwords(r))==k]
        out.append((np.array([r[scorer]['nll'] for r in rs]),np.array([r[scorer]['tokens'] for r in rs])))
    return out

def band_counts(rows,scorer): return [len(a) for a,_ in band_arrays(rows,scorer)]

def support(counts):
    if min(counts)==0: return 'not_estimable'
    return 'ok' if min(counts)>=MIN_BAND else 'insufficient_support'

def standardised(arrays): return math.exp(float(sum(w*a.sum()/t.sum() for w,(a,t) in zip(WEIGHTS,arrays))))

def ppl(rows,scorer):
    s=scored(rows,scorer); counts=band_counts(rows,scorer); status=support(counts)
    return dict(scored=len(s),raw=math.exp(sum(r[scorer]['nll'] for r in s)/sum(r[scorer]['tokens'] for r in s)) if s else None,
                band_counts=counts,support=status,standardised=standardised(band_arrays(rows,scorer)) if status=='ok' else None)

def band_bootstrap(rows,scorer,seed,reps=10000,log=True):
    """Within-band row bootstrap keeping band counts and fixed weights; returns replicate (log) standardised perplexities."""
    rng=np.random.default_rng(seed); out=np.zeros(reps)
    for w,(nll,tok) in zip(WEIGHTS,band_arrays(rows,scorer)):
        idx=rng.integers(0,len(nll),size=(reps,len(nll)))
        out+=w*nll[idx].sum(1)/tok[idx].sum(1)
    return out if log else np.exp(out)

# ---------------------------------------------------------------- diversity and overlap (Phase A section 5, plus 4-grams)
def ngrams(text,n): t=norm(text).split(); return [' '.join(t[i:i+n]) for i in range(len(t)-n+1)]

def train_sets(train_rows):
    return {norm(r['text']) for r in train_rows},{g for r in train_rows for g in ngrams(r['text'],4)},{g for r in train_rows for g in ngrams(r['text'],8)}

def diversity(rows,train):
    texts,g4,g8=train
    words=[r['text'].split() for r in rows]
    bigrams=[tuple(w[i:i+2]) for w in words for i in range(len(w)-1)]
    uni=Counter(t for r in rows for t in norm(r['text']).split()); total=sum(uni.values())
    four=[[g in g4 for g in ngrams(r['text'],4)] for r in rows]
    eight=[g in g8 for r in rows for g in ngrams(r['text'],8)]
    shares=np.array([np.mean(f) for f in four if f])
    pooled4=[x for f in four for x in f]
    return dict(n=len(rows),mean_words=float(np.mean([len(w) for w in words])),
        distinct2=len(set(bigrams))/max(1,len(bigrams)),
        unigram_entropy_bits=float(-sum(c/total*math.log2(c/total) for c in uni.values())) if total else None,
        training_copies=sum(norm(r['text']) in texts for r in rows),
        fourgram_overlap=float(np.mean(pooled4)) if pooled4 else None,fourgram_total=len(pooled4),
        fourgram_row_share_quartiles=np.percentile(shares,[25,50,75]).tolist() if len(shares) else None,
        high_overlap_rows=sum(len(f)>=4 and np.mean(f)>=.9 for f in four),rows_under_4_words=sum(len(w)<4 for w in words),
        eightgram_overlap=float(np.mean(eight)) if eight else None,eightgram_total=len(eight),
        empty=sum(not r['text'].strip() for r in rows))

# ---------------------------------------------------------------- dictionary tokens (section 4)
TOKEN=re.compile(r"(?:['’][nN])(?![^\W\d_])|ŉ|[^\W\d_]+(?:['’\-][^\W\d_]+)*")
ARTICLES={"'n","’n","'N","’N",'ŉ'}

def dictionary_tokens(text):
    """Yields (surface, kind) with kind in article, acronym, word."""
    for m in TOKEN.finditer(unicodedata.normalize('NFC',text)):
        s=m.group(0)
        if s in ARTICLES: yield s,'article'
        elif len(re.sub(r'[^\w]','',s))>=2 and s.isupper(): yield s,'acronym'
        else: yield s,'word'

# ---------------------------------------------------------------- interval helpers
def wilson(k,n,z=1.959964):
    if n==0: return None
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def newcombe(k1,n1,k2,n2):
    """Newcombe hybrid score interval for p1-p2."""
    p1,p2=k1/n1,k2/n2; (l1,u1),(l2,u2)=wilson(k1,n1),wilson(k2,n2); d=p1-p2
    return [d-math.sqrt((p1-l1)**2+(u2-p2)**2),d+math.sqrt((u1-p1)**2+(p2-l2)**2)]

def welch(a,b):
    from scipy import stats
    a,b=np.asarray(a),np.asarray(b); va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)); half=stats.t.ppf(.975,df)*math.sqrt(va+vb); d=float(a.mean()-b.mean())
    return dict(estimate=d,low=d-half,high=d+half,df=float(df))

def status(low,high): return 'below_0' if high<0 else 'above_0' if low>0 else 'contains_0'

def doc_bootstrap(rows,stat,seed,reps=2000):
    by=defaultdict(list)
    for r in rows: by[r['document_id']].append(r)
    docs=sorted(by); rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        pick=rng.integers(0,len(docs),len(docs)); vals.append(stat([r for i in pick for r in by[docs[i]]]))
    return np.percentile(vals,[2.5,97.5]).tolist()

def mean_sd(vals):
    v=np.array([x for x in vals if x is not None],dtype=float)
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),values=v.tolist()) if len(v)==len(vals) and len(v)>1 else dict(values=list(vals))

# ---------------------------------------------------------------- final analysis
def bank_files():
    """Bank name -> merged per-row records (text, scores, part A rows, part B handled from text)."""
    return sorted(p.stem[len('bank-'):] for p in OUT.glob('bank-*.jsonl'))

def load_bank(name):
    rows=rows_of(OUT/f'bank-{name}.jsonl')
    for part in ('lid','dict','goldfish'):
        p=OUT/f'{part}-{name}.jsonl'
        if p.exists():
            extra=rows_of(p); assert len(extra)==len(rows),(part,name)
            for r,e in zip(rows,extra):
                if part=='goldfish': r['goldfish']=e['goldfish']
                else: r[part]=e
    return rows

def part_a(rows):
    out={}
    lid=[r['lid'] for r in rows if r.get('lid') and r['lid'].get('top')]
    if lid:
        n=len(lid); out['lid']=dict(n=n,empty=len(rows)-n,
            afr_top=sum(x['top']=='__label__afr_Latn' for x in lid)/n,nld_top=sum(x['top']=='__label__nld_Latn' for x in lid)/n,
            other_top=sum(x['top'] not in ('__label__afr_Latn','__label__nld_Latn') for x in lid)/n,
            mean_p_afr=float(np.mean([x['p_afr'] for x in lid])),
            by_band=[dict(n=len(b),afr_top=sum(x['top']=='__label__afr_Latn' for x in b)/len(b) if b else None,
                          nld_top=sum(x['top']=='__label__nld_Latn' for x in b)/len(b) if b else None)
                     for k in range(4) for b in [[r['lid'] for r in rows if r.get('lid') and r['lid'].get('top') and nwords(r)>0 and band(nwords(r))==k]]])
    d=[r['dict'] for r in rows if r.get('dict')]
    if d:
        checked=sum(x['checked'] for x in d); unrec=sum(x['unrecognised'] for x in d); acr=sum(x['acronyms'] for x in d); acr_unrec=sum(x['acronyms_unrecognised'] for x in d)
        types=defaultdict(list)
        for x in d:
            for s,ok in x['words']: types[unicodedata.normalize('NFC',s).casefold()].append(ok)
        unrec_types=[t for t,v in types.items() if not any(v)]
        freq=Counter(unicodedata.normalize('NFC',s).casefold() for x in d for s,ok in x['words'] if not ok)
        out['dictionary']=dict(checked_tokens=checked,unrecognised_tokens=unrec,token_rate=unrec/checked if checked else None,
            checked_types=len(types),type_rate=len(unrec_types)/len(types) if types else None,
            acronyms=acr,acronym_fraction=acr/(checked+acr) if checked+acr else None,
            token_rate_with_acronyms=(unrec+acr_unrec)/(checked+acr) if checked+acr else None,
            apostrophe_replacements=sum(x['apostrophe_replacements'] for x in d),
            top_unrecognised=freq.most_common(20))
    return out

def token_rate(rows):
    c=sum(r['dict']['checked'] for r in rows); return sum(r['dict']['unrecognised'] for r in rows)/c if c else float('nan')

def afr_share(rows):
    l=[r['lid'] for r in rows if r['lid'].get('top')]; return sum(x['top']=='__label__afr_Latn' for x in l)/len(l)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    train=train_sets(load_train())
    names=bank_files(); banks={n:load_bank(n) for n in names}
    res=dict(protocol_sha256_lf=read(OUT/'contract.json')['protocol_sha256_lf'],banks={})
    for n,rows in banks.items():
        b=dict(part_a=part_a(rows),part_b=diversity(rows,train))
        for s in ('mgpt','goldfish'):
            if any(r.get(s) for r in rows): b[s]=ppl(rows,s)
        res['banks'][n]=b
    hl=banks['HL']; hl_rate=res['banks']['HL']['part_a']['dictionary']['token_rate']
    for r in hl: f=[g in train[1] for g in ngrams(r['text'],4)]; r['_g4']=(sum(f),len(f))
    res['HL_intervals']=dict(token_rate=doc_bootstrap(hl,token_rate,20261103),afr_top=doc_bootstrap(hl,afr_share,20261103),
        fourgram_overlap=doc_bootstrap(hl,lambda rs: sum(r['_g4'][0] for r in rs)/sum(r['_g4'][1] for r in rs),20261103),
        note='2,000 document-bootstrap replicates, seed 20261103')
    for n,b in res['banks'].items():
        d=b['part_a'].get('dictionary')
        if d: d['excess_over_HL']=d['token_rate']-hl_rate
    res['diffusion_summary']={size:{k:mean_sd([f(res['banks'][r]) for r in runs]) for k,f in {
        'afr_top':lambda b:b['part_a']['lid']['afr_top'],'token_rate':lambda b:b['part_a']['dictionary']['token_rate'],
        'excess_over_HL':lambda b:b['part_a']['dictionary']['excess_over_HL'],'fourgram_overlap':lambda b:b['part_b']['fourgram_overlap'],
        'high_overlap_rows':lambda b:b['part_b']['high_overlap_rows'],'goldfish_std':lambda b:b['goldfish']['standardised'],
        'mgpt_std':lambda b:b['mgpt']['standardised']}.items()} for size,runs in SIZES.items()}
    res['part_c']=part_c(banks,res)
    res['part_d']=part_d(banks,res)
    for name in ('mauve','judge-selection','judge-application'):
        p=OUT/f'{name}.json'
        res[name.replace('-','_')]=read(p) if p.exists() else 'not available'
    (OUT/'analysis.json').write_text(json.dumps(res,indent=1,ensure_ascii=False,default=float)+'\n',encoding='utf8')
    frontier(res)
    print('analysis written')

def part_c(banks,res):
    out={}
    for s in ('mgpt','goldfish'):
        logs={r:math.log(res['banks'][r][s]['standardised']) for r in RUNS if res['banks'][r][s]['standardised']}
        c={}
        c['C-1']=welch([logs[r] for r in SIZES['large']],[logs[r] for r in SIZES['small']]) if len(logs)==10 else 'insufficient_support'
        ar=banks['AR-hist']
        if res['banks']['AR-hist'][s]['support']=='ok':
            v=float(np.var(band_bootstrap(ar,s,20261102),ddof=1)); lar=math.log(res['banks']['AR-hist'][s]['standardised'])
            for size,runs in SIZES.items():
                x=np.array([logs[r] for r in runs]); d=float(x.mean()-lar); h=T4_9875*math.sqrt(x.var(ddof=1)/5+v)
                c[f'C-2{size[0]}']=dict(estimate=d,low=d-h,high=d+h,ratio=math.exp(d),ratio_low=math.exp(d-h),ratio_high=math.exp(d+h))
        c3=banks['C3']
        if s=='mgpt':
            pa=read(PHASE_A/'analysis.json')['primary']
            for size in SIZES: c[f'C-3{size[0]}']={k:pa[size][k] for k in ('difference','low','high','classification') if k in pa[size]}
        elif res['banks']['C3'][s]['support']=='ok':
            v=float(np.var(band_bootstrap(c3,s,20261030,log=False),ddof=1)); base=res['banks']['C3'][s]['standardised']
            for size,runs in SIZES.items():
                x=np.array([res['banks'][r][s]['standardised'] for r in runs]); d=float(x.mean()-base); h=T4_9875*math.sqrt(x.var(ddof=1)/5+v)
                c[f'C-3{size[0]}']=dict(difference=d,low=d-h,high=d+h)
        out[s]=c
    comp={}
    for k in out['mgpt']:
        a,b=out['mgpt'].get(k),out['goldfish'].get(k)
        if not isinstance(a,dict) or not isinstance(b,dict) or 'low' not in a or 'low' not in b: comp[k]='insufficient_support'; continue
        ea,eb=a.get('estimate',a.get('difference')),b.get('estimate',b.get('difference'))
        sa,sb=status(a['low'],a['high']),status(b['low'],b['high'])
        comp[k]=dict(direction='same' if np.sign(ea)==np.sign(eb) else 'opposite',mgpt_interval=sa,goldfish_interval=sb,
                     confirmed_under_both=np.sign(ea)==np.sign(eb) and sa==sb!='contains_0')
    out['comparison']=comp
    from scipy.stats import spearmanr
    both=[n for n,b in res['banks'].items() if b.get('mgpt',{}).get('standardised') and b.get('goldfish',{}).get('standardised')]
    rho=spearmanr([res['banks'][n]['mgpt']['standardised'] for n in both],[res['banks'][n]['goldfish']['standardised'] for n in both])
    out['spearman']=dict(rho=float(rho.statistic),banks=len(both))
    texts={norm(r['text']) for r in load_train()}
    out['copies_removed']={n:{s:dict(removed=sum(norm(r['text']) in texts for r in banks[n]),**ppl([r for r in banks[n] if norm(r['text']) not in texts],s))
                              for s in ('mgpt','goldfish')} for n in RUNS+['AR-hist']}
    return out

def part_d(banks,res):
    sel=OUT/'ar-tuned-selection.json'
    if not sel.exists(): return 'not available'
    sel=read(sel); out=dict(selection=sel)
    if sel.get('tuned') is None: return out
    fresh=banks['AR-fresh']; fb=res['banks']['AR-fresh']
    out['fresh']=dict(mgpt=fb['mgpt'],distinct2=fb['part_b']['distinct2'],entropy=fb['part_b']['unigram_entropy_bits'],
        copies=fb['part_b']['training_copies'],still_eligible=eligible(fb,sel['floors']))
    if fb['mgpt']['support']!='ok': out['primary']='insufficient_support'; return out
    v=float(np.var(band_bootstrap(fresh,'mgpt',20261104),ddof=1)); lf=math.log(fb['mgpt']['standardised']); prim={}
    for size,runs in SIZES.items():
        x=np.array([math.log(res['banks'][r]['mgpt']['standardised']) for r in runs]); d=float(x.mean()-lf); h=T4_9875*math.sqrt(x.var(ddof=1)/5+v)
        lo,hi=math.exp(d-h),math.exp(d+h)
        prim[size]=dict(ratio=math.exp(d),low=lo,high=hi,
            classification='meets' if hi<=CRITERION else 'does_not_meet' if lo>CRITERION else 'not_established',
            relative_to_1='lower' if hi<1 else 'higher' if lo>1 else 'not_different_under_this_procedure')
    out['primary']=prim
    cell=res['banks'][f"AR-grid-{sel['tuned']}"]['mgpt']['standardised']
    out['selection_sample_ratio_no_interval']={size:math.exp(np.mean([math.log(res['banks'][r]['mgpt']['standardised']) for r in runs])-math.log(cell)) for size,runs in SIZES.items()}
    return out

def eligible(b,floors):
    return b['mgpt']['support']=='ok' and b['part_b']['distinct2']>=floors['distinct2'] and b['part_b']['unigram_entropy_bits']>=floors['entropy'] and b['part_b']['training_copies']<=floors['copies']

def frontier(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,label in ((axes[0],'unigram_entropy_bits','Unigram entropy (bits)'),(axes[1],'distinct2','Distinct-2')):
        for n,b in res['banks'].items():
            y=b.get('mgpt',{}).get('standardised')
            if y is None or n in ('HL',): continue
            x=b['part_b'][key]
            if n.startswith('AR-grid-'): ax.scatter(x,y,marker='o' if n.endswith('off') else 's',facecolors='none',edgecolors='C0',s=28)
            elif n=='AR-fresh': ax.scatter(x,y,marker='*',color='C0',s=110,label='AR tuned (fresh bank)')
            elif n.startswith('small'): ax.scatter(x,y,color='C1',s=22)
            elif n.startswith('published'): ax.scatter(x,y,color='C3',s=22)
            elif n in ('C1','C2','C3'): ax.scatter(x,y,color='grey',marker='x'); ax.annotate(n,(x,y),fontsize=7)
            elif n=='H': ax.scatter(x,y,color='k',marker='D',s=18); ax.annotate('H (811 rows)',(x,y),fontsize=7)
        ax.set_xlabel(label); ax.set_ylabel('Standardised mGPT perplexity')
    axes[0].set_title('AR grid: circles constraints off, squares on; orange small, red large diffusion',fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'frontier.pdf'); plt.close(fig)

if __name__=='__main__':
    main()
