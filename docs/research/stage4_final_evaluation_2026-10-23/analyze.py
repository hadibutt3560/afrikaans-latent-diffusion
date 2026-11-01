"""Secondary analysis of frozen outputs; no model selection."""
import json,hashlib,math,csv,unicodedata
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; EVAL=ROOT/'outputs/longrun-local-evaluation'
OUT=ROOT/'outputs/stage4-final-evaluation-20261023'
rng=np.random.default_rng(20261023); B=10000; sources={}
def read(p):
 p=Path(p); sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):
 p=Path(p); sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def norm(t): return ' '.join(unicodedata.normalize('NFC',t).casefold().split())
train={norm(r['text']) for r in lines(ROOT/'data/nchlt_afrikaans/processed/stage_5_7/train.jsonl')}
def ppl(rs): return math.exp(sum(r['mgpt']['nll'] for r in rs)/sum(r['mgpt']['tokens'] for r in rs))
def bootstrap(rs):
 a=np.array([[r['mgpt']['nll'],r['mgpt']['tokens']] for r in rs]); out=[]
 assert np.isfinite(a).all() and (a[:,1]>0).all()
 for start in range(0,B,100):
  s=a[rng.integers(0,len(a),(min(100,B-start),len(a)))].sum(1); out.extend(np.exp(s[:,0]/s[:,1]))
 return np.array(out)
def metrics(rs):
 words=[r['text'].split() for r in rs]; normalized=[norm(r['text']) for r in rs]
 out=dict(n=len(rs),ppl=ppl(rs),mean_words=float(np.mean(list(map(len,words)))),empty=sum(not t for t in normalized),duplicate_excess=len(rs)-len(set(normalized)),train_matches=sum(t in train for t in normalized),eos=sum(r.get('eos',False) for r in rs),cap=sum(r.get('length_cap',False) for r in rs),sentinel=sum(r.get('sentinel',False) for r in rs))
 for n in (1,2,3,4):
  counts=[Counter(tuple(w[i:i+n]) for i in range(max(0,len(w)-n+1))) for w in words]
  total=Counter()
  for c in counts: total.update(c)
  out[f'distinct_{n}']=len(total)/sum(total.values()) if total else 0
  if n>=3: out[f'mean_repeat_{n}']=float(np.mean([1-len(c)/sum(c.values()) if c else 0 for c in counts]))
 return out
run_table=[]; banks={}; boots={}; bands={}
for family in ('small','published'):
 for seed in range(42,47):
  name=f'{family}-seed{seed}'; folder=EVAL/'runs'/name
  scored=read(folder/'test-mgpt.json'); rs=scored['rows']; assert len(rs)==1000 and len({r['seed'] for r in rs})==1000
  assert abs(ppl(rs)-scored['token_weighted_perplexity'])<1e-8
  final=read(folder/'final-test.json'); val=read(folder/'selected-validation.json'); status=read(folder/'status.json')
  assert final['selected_step']==status['best_step']
  sampleboot=bootstrap(rs); boots[name]=sampleboot
  run_table.append(dict(run=name,family=family,seed=seed,selected_step=final['selected_step'],test_mse=final['denoising']['test_loss'],valid_mse=val['denoising']['valid_loss'],hours=status['cumulative_seconds']/3600,**metrics(rs),sample_ci_low=float(np.quantile(sampleboot,.025)),sample_ci_high=float(np.quantile(sampleboot,.975))))
  banks[name]=rs
  v=read(folder/'validation-mgpt.json')['rows'][:256]; assert len(v)==256
  banks['validation/'+name]=v; boots['validation/'+name]=bootstrap(v)
ar=lines(ROOT/'outputs/ar-validation-audit-20261022/ar-cap127-scored.jsonl'); assert len(ar)==256
banks['validation/AR']=ar; boots['validation/AR']=bootstrap(ar)
valid_table=[dict(run=name,**metrics(rs),sample_ci_low=float(np.quantile(boots[name],.025)),sample_ci_high=float(np.quantile(boots[name],.975))) for name,rs in banks.items() if name.startswith('validation/')]
summaries={}
for f in ('small','published'):
 rows=[r for r in run_table if r['family']==f]
 summaries[f]={k:dict(mean=float(np.mean([r[k] for r in rows])),sd=float(np.std([r[k] for r in rows],ddof=1)),min=min(r[k] for r in rows),max=max(r[k] for r in rows)) for k in ('ppl','test_mse','valid_mse','distinct_1','distinct_2','mean_repeat_3','mean_repeat_4','mean_words')}
 summaries[f]['counts']={k:sum(r[k] for r in rows) for k in ('n','empty','duplicate_excess','train_matches','eos','cap','sentinel')}
small=np.array([r['ppl'] for r in run_table if r['family']=='small']); large=np.array([r['ppl'] for r in run_table if r['family']=='published'])
diff=large[rng.integers(0,5,(B,5))].mean(1)-small[rng.integers(0,5,(B,5))].mean(1)
contrast=dict(large_minus_small=float(large.mean()-small.mean()),relative_percent=float((large.mean()/small.mean()-1)*100),unpaired_training_bootstrap_95=np.quantile(diff,[.025,.975]).tolist())
ratios={}
for f in ('small','published'):
 names=[f'validation/{f}-seed{s}' for s in range(42,47)]
 means=np.array([ppl(banks[n]) for n in names]); arrays=np.stack([boots[n] for n in names])
 picks=rng.integers(0,5,(B,5)); reps=rng.integers(0,B,(B,5)); sample=arrays[picks,reps].mean(1)/boots['validation/AR']
 ratios[f]=dict(arithmetic_mean_ratio=float(means.mean()/ppl(ar)),geometric_mean_ratio=float(np.exp(np.log(means).mean())/ppl(ar)),conditional_hierarchical_95=np.quantile(sample,[.025,.975]).tolist(),conditional_one_sided_upper95=float(np.quantile(sample,.95)),limitation='Conditional on single historical AR and frozen AE; not matched-family noninferiority proof.')
for name in ('test-reference-mgpt','validation-reference-mgpt'):
 rs=read(EVAL/(name+'.json'))['rows']; banks[name]=rs
for name,rs in banks.items():
 bands[name]={}
 for label,lo,hi in [('1-10',1,10),('11-20',11,20),('21-35',21,35),('36+',36,100000)]:
  sub=[r for r in rs if lo<=len(r['text'].split())<=hi]
  bands[name][label]=dict(n=len(sub),ppl=ppl(sub) if sub else None)
scores=lines(OUT/'reconstruction-scores.jsonl'); original=lines(ROOT/'outputs/reconstruction-audit-20261022/examples.jsonl')
assert [r['id'] for r in scores]==[r['id'] for r in original]
recon={k:math.exp(sum(r[k]['nll'] for r in scores)/sum(r[k]['tokens'] for r in scores)) for k in ('source_mgpt','reconstruction_mgpt')}
docs=sorted({r['document_id'] for r in scores}); a=np.array([[sum(r[k][v] for r in scores if r['document_id']==d) for k in ('source_mgpt','reconstruction_mgpt') for v in ('nll','tokens')] for d in docs])
s=a[rng.integers(0,len(a),(B,len(a)))].sum(1); recon['log_ppl_difference_95']=np.quantile(s[:,2]/s[:,3]-s[:,0]/s[:,1],[.025,.975]).tolist()
fert=[r['fertility'] for r in original]
associations={k:float(spearmanr(fert,v).statistic) for k,v in dict(reconstruction_ce=[r['correct']['ce'] for r in original],reconstruction_external_ppl=[math.exp(r['reconstruction_mgpt']['nll']/r['reconstruction_mgpt']['tokens']) for r in scores],source_token_length=[r['full_tokens'] for r in original]).items()}
ae={split:read(ROOT/('outputs/cluster-results/afrikaans-'+split+'1024-20260919-'+('144340-364831' if split=='valid' else '172636-212013'))/f'autoencoder-{split}-step-30500.json') for split in ('valid','test')}
tokenizer=read(ROOT/'data/nchlt_afrikaans/analysis/mt5-base-tokenizer/report.json')
result=dict(test_summary=summaries,capacity_contrast=contrast,validation_ar_ratios=ratios,reconstruction_external=recon,fertility_spearman_exploratory=associations,historical_ae=ae,real_text_bands={k:v for k,v in bands.items() if 'reference' in k},human_evaluation='Not conducted; explicitly outside submitted proposal and user reaffirmed scope.')
for name,value in [('analysis.json',result),('length_bands.json',bands),('tokenizer_report_copy.json',tokenizer),('source_hashes.json',sources)]: (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
for name,rows in [('test_seed_table.csv',run_table),('validation_comparison.csv',valid_table)]:
 with (HERE/name).open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(HERE/'checks.json').write_text(json.dumps(dict(status='passed',test_runs=10,test_outputs=10000,validation_comparison_runs=11,reconstruction_pairs=81,stored_test_ppl_recomputed=True,source_ids_aligned=True,analysis_seed=20261023,bootstrap_replicates=B),indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
