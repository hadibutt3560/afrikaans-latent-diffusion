"""Summarise a completed train-versus-held-out run and apply the PROTOCOL.md thresholds.

Usage: python analyze.py outputs/train-heldout-recon-YYYYMMDD
No inference, tuning or threshold changes.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_diagnostic import digest, lf_digest, PROTOCOL_SHA, AE_SHA, TRAIN_SHA, VALID_SHA
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools')); from as_run import as_run_sha256, as_run_matches  # protocols are checked as they were at tag as-run
BOOTSTRAP_SEED=91431
REPLICATES=10000
BINS=('short','medium','long')

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def rows_of(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]

def check(out,manifest,contract,rows):
    assert as_run_sha256(HERE/'PROTOCOL.md')==PROTOCOL_SHA==contract['protocol_sha256_lf']
    assert digest(HERE/'run_diagnostic.py')==contract['script_sha256'], 'Runner changed after the run'
    assert digest(out/'manifest.json')==contract['manifest_sha256']
    assert (contract['ae_sha256'],contract['train_sha256'],contract['valid_sha256'])==(AE_SHA,TRAIN_SHA,VALID_SHA)
    anchor=rows_of(out/'examples-anchor.jsonl')
    assert len(anchor)==len(manifest['anchor']) and all(a['reproduced'] for a in anchor), 'G1 not passed'
    for name in ('T','V'):
        rs=rows[name]; panel=manifest[name]; donors=manifest['donor_indices'][name]
        assert [r['id'] for r in rs]==[p['id'] for p in panel], f'Panel {name} incomplete or reordered'
        for i,r in enumerate(rs):
            d=panel[donors[str(i)]]
            assert d['id']==r['donor_id'] and d['document_id']!=r['document_id'] and d['bin']==r['bin']
            # Decoding depends only on the latent, so the shuffled output must equal the donor's own output.
            assert r['shuffled']['beam']['generated_ids']==rs[donors[str(i)]]['correct']['beam']['generated_ids']
            for arm in ('correct','shuffled'):
                assert np.isfinite(r[arm]['nll']) and r[arm]['tokens']==len(panel[i]['ids'])
    assert not ({r['document_id'] for r in rows['T']}&{r['document_id'] for r in rows['V']})

def summary(rs,arm,decoding='beam',view='model_metrics'):
    if not rs: return None
    tokens=sum(r[arm]['tokens'] for r in rs)
    m=[r[arm][decoding][view] for r in rs]
    expected=sum(x['number_expected'] for x in m); predicted=sum(x['number_predicted'] for x in m); matched=sum(x['number_matched'] for x in m)
    return dict(n=len(rs),documents=len({r['document_id'] for r in rs}),tokens=tokens,ce=sum(r[arm]['nll'] for r in rs)/tokens,
        accuracy=sum(r[arm]['correct_tokens'] for r in rs)/tokens,exact_count=sum(x['exact'] for x in m),exact_rate=sum(x['exact'] for x in m)/len(rs),
        chrf=float(np.mean([x['chrf'] for x in m])),similarity=float(np.mean([x['similarity'] for x in m])),
        number_expected=expected,number_recall=matched/expected if expected else None,number_precision=matched/predicted if predicted else None,
        eos_count=sum(r[arm][decoding]['eos'] for r in rs),cap_count=sum(r[arm][decoding]['at_cap'] for r in rs))

def document_table(rs):
    """Additive per-document statistics, so a document resample is a row resample."""
    docs=sorted({r['document_id'] for r in rs}); table=[]
    for d in docs:
        ds=[r for r in rs if r['document_id']==d]
        row=[len(ds),sum(r['correct']['beam']['model_metrics']['exact'] for r in ds),sum(r['correct']['nll'] for r in ds),
            sum(r['correct']['tokens'] for r in ds),sum(r['correct']['beam']['model_metrics']['chrf'] for r in ds)]
        for b in BINS:
            bs=[r for r in ds if r['bin']==b]
            row+=[len(bs),sum(r['correct']['beam']['model_metrics']['exact'] for r in bs)]
        table.append(row)
    return np.array(table,dtype=float)

def statistics(t,v,weights):
    """Columns as in document_table; rows are panels (or bootstrap replicates summed over documents)."""
    with np.errstate(divide='ignore',invalid='ignore'):
        e_t,e_v=t[...,1]/t[...,0],v[...,1]/v[...,0]
        out=dict(delta_exact=e_t-e_v,ratio_exact=e_t/e_v,delta_ce=v[...,2]/v[...,3]-t[...,2]/t[...,3],delta_chrf=t[...,4]/t[...,0]-v[...,4]/v[...,0])
        reweighted=0
        for k,b in enumerate(BINS):
            n,x=5+2*k,6+2*k
            if weights[b]: reweighted=reweighted+weights[b]*(t[...,x]/t[...,n]-v[...,x]/v[...,n])
        out['delta_exact_bin_reweighted']=reweighted
    return out

def interval(values):
    finite=values[np.isfinite(values)]
    lo,hi=np.quantile(finite,[.025,.975],method='inverted_cdf')
    return dict(low=float(lo),high=float(hi),non_finite_replicates=int(values.size-finite.size))

def classify(point,ci,t_shuffled,v_shuffled):
    if not (t_shuffled['exact_rate']<=.02 and t_shuffled['chrf']-v_shuffled['chrf']<=5): return 'M'
    e_t,e_v=point['exact_T'],point['exact_V']
    exact_lo,exact_hi=ci['delta_exact']['low'],ci['delta_exact']['high']
    if e_t>=3*e_v and exact_lo>=.10 and ci['delta_ce']['low']>0: return 'G'
    if e_t<=1.5*e_v and exact_hi<=.10: return 'F'
    return 'I'

def main():
    out=Path(sys.argv[1]).resolve()
    manifest,contract=read(out/'manifest.json'),read(out/'contract.json')
    if (out/'failure.json').exists(): raise SystemExit(f"Run stopped at a gate: {read(out/'failure.json')}")
    rows={k:rows_of(out/f'examples-{k}.jsonl') for k in ('T','V')}
    check(out,manifest,contract,rows)
    weights={b:sum(r['bin']==b for r in rows['V'])/len(rows['V']) for b in BINS}
    tables={k:document_table(rows[k]) for k in ('T','V')}
    point=statistics(tables['T'].sum(0),tables['V'].sum(0),weights)
    point={k:float(v) for k,v in point.items()}
    point.update(exact_T=tables['T'][:,1].sum()/tables['T'][:,0].sum(),exact_V=tables['V'][:,1].sum()/tables['V'][:,0].sum())
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    bt=tables['T'][rng.integers(0,len(tables['T']),(REPLICATES,len(tables['T'])))].sum(1)
    bv=tables['V'][rng.integers(0,len(tables['V']),(REPLICATES,len(tables['V'])))].sum(1)
    ci={k:interval(np.asarray(v,dtype=float)) for k,v in statistics(bt,bv,weights).items()}
    arms={k:dict(correct_beam=summary(rows[k],'correct'),shuffled_beam=summary(rows[k],'shuffled'),correct_greedy=summary(rows[k],'correct','greedy'),
        correct_beam_source_view=summary(rows[k],'correct','beam','source_metrics')) for k in ('T','V')}
    bins={k:{b:dict(correct_beam=summary([r for r in rows[k] if r['bin']==b],'correct'),shuffled_beam=summary([r for r in rows[k] if r['bin']==b],'shuffled')) for b in BINS} for k in ('T','V')}
    outcome=classify(point,ci,arms['T']['shuffled_beam'],arms['V']['shuffled_beam'])
    report=dict(outcome=outcome,point=point,bootstrap_95=ci,bootstrap=dict(seed=BOOTSTRAP_SEED,replicates=REPLICATES,unit='document, independently within each panel'),
        bin_weights_from_V=weights,arms=arms,length_bins=bins,
        gate_G3=dict(shuffled_exact_T=arms['T']['shuffled_beam']['exact_rate'],shuffled_chrf_T_minus_V=arms['T']['shuffled_beam']['chrf']-arms['V']['shuffled_beam']['chrf']),
        note='Thresholds from PROTOCOL.md section 7; not to be changed after this report.')
    with (out/'analysis.json').open('x',encoding='utf8') as f: json.dump(report,f,indent=2,allow_nan=False)
    lines=['# All reconstructions','','AI-assisted review material; not human evaluation. Order is manifest order.','']
    for k in ('T','V'):
        for i,r in enumerate(rows[k]):
            lines+=[f"## {k}{i}: {r['id']} ({r['bin']})",f"Source: {r['source']}",f"Model view: {r['model_view']}",
                f"Correct, beam: {r['correct']['beam']['text']}",f"Correct, greedy: {r['correct']['greedy']['text']}",f"Shuffled, beam: {r['shuffled']['beam']['text']}",'']
    with (out/'ALL_RECONSTRUCTIONS.md').open('x',encoding='utf8') as f: f.write('\n\n'.join(lines))
    print(json.dumps(dict(outcome=outcome,point=point,bootstrap_95=ci),indent=2))

if __name__=='__main__': main()
