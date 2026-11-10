"""Summarise only the fixed Step 1 diagnostic; no inference or tuning."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/reconstruction-audit-20261022'
rows=[json.loads(s) for s in (OUT/'examples.jsonl').read_text(encoding='utf8').splitlines()]
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
contract=json.loads((OUT/'contract.json').read_text(encoding='utf8'))
assert len(rows)==contract['selected']
assert [r['id'] for r in rows]==[r['id'] for r in manifest['selected']]
assert not ({r['document_id'] for r in rows}&{r['document_id'] for r in manifest['reserved']})
import sys; sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools')); from as_run import as_run_sha256, as_run_matches  # protocols are checked as they were at tag as-run
assert as_run_matches(HERE/'PROTOCOL.md',contract['protocol_sha256']), 'PROTOCOL.md at tag as-run differs from the run record'
assert hashlib.sha256((HERE/'run_audit.py').read_bytes()).hexdigest()==contract['script_sha256']
from run_audit import digest, BUNDLE, AE, DATA
assert digest(AE/'model.pt')==contract['ae_sha256'] and digest(DATA)==contract['data_sha256']
original=json.loads((BUNDLE/'inputs.json').read_text(encoding='utf8'))
for name,h in original['model_cache'].items(): assert digest(BUNDLE/'models/mt5-base'/name)==h,name
for i,r in enumerate(rows):
    d=manifest['selected'][manifest['donor_indices'][str(i)]]
    assert d['id']==r['donor_id'] and d['document_id']!=r['document_id'] and d['bin']==r['bin']
    assert r['shuffled']['generated_ids']==rows[manifest['donor_indices'][str(i)]]['correct']['generated_ids']
    for arm in ('correct','shuffled'):
        a=r[arm]
        assert np.isfinite(a['nll']) and a['tokens']==len(manifest['selected'][i]['ids'])
        assert a['eos']==(1 in a['generated_ids']) and a['at_cap']==(len(a['generated_ids'])>=128)
def summary(rs,arm,view='model_metrics'):
    n=sum(r[arm]['tokens'] for r in rs)
    expected=sum(r[arm][view]['number_expected'] for r in rs)
    predicted=sum(r[arm][view]['number_predicted'] for r in rs)
    matched=sum(r[arm][view]['number_matched'] for r in rs)
    return dict(n=len(rs),tokens=n,ce=sum(r[arm]['nll'] for r in rs)/n,accuracy=sum(r[arm]['correct_tokens'] for r in rs)/n,
        exact_count=sum(r[arm][view]['exact'] for r in rs),similarity=np.mean([r[arm][view]['similarity'] for r in rs]),chrf=np.mean([r[arm][view]['chrf'] for r in rs]),
        number_expected=expected,number_predicted=predicted,number_matched=matched,number_recall=matched/expected if expected else None,number_precision=matched/predicted if predicted else None,
        eos_count=sum(r[arm]['eos'] for r in rs),cap_count=sum(r[arm]['at_cap'] for r in rs))
docs=sorted({r['document_id'] for r in rows})
# Additive document statistics preserve paired arms and variable passage counts.
stats=[]
for d in docs:
    rs=[r for r in rows if r['document_id']==d]
    stats.append([sum(r['correct']['tokens'] for r in rs),sum(r['shuffled']['nll']-r['correct']['nll'] for r in rs),sum(r['correct']['correct_tokens']-r['shuffled']['correct_tokens'] for r in rs),len(rs),sum(r['correct']['model_metrics']['chrf']-r['shuffled']['model_metrics']['chrf'] for r in rs),sum(r['correct']['model_metrics']['similarity']-r['shuffled']['model_metrics']['similarity'] for r in rs)])
a=np.array(stats); rng=np.random.default_rng(91428)
b=a[rng.integers(0,len(docs),(10000,len(docs)))].sum(1)
cis={name:np.quantile(b[:,num]/b[:,den],[.025,.975]).tolist() for name,num,den in [('ce_shuffled_minus_correct',1,0),('accuracy_correct_minus_shuffled',2,0),('chrf_correct_minus_shuffled',4,3),('similarity_correct_minus_shuffled',5,3)]}
report=dict(arms={arm:summary(rows,arm) for arm in ('correct','shuffled')},source_view={arm:summary(rows,arm,'source_metrics') for arm in ('correct','shuffled')},document_bootstrap_95=cis,length_bins={label:{arm:summary([r for r in rows if r['bin']==label],arm) for arm in ('correct','shuffled')} for label in sorted({r['bin'] for r in rows})},documents=len(docs),input_view_changed=sum(r['source']!=r['model_view'] for r in rows),truncated=sum(r['full_tokens']>128 for r in rows))
(OUT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
lines=['# All diagnostic reconstructions','', 'AI-assisted review material; no fluent-human ratings. Numbers denote fixed manifest order. Shuffled donor IDs and full scores are in examples.jsonl.','']
for i,r in enumerate(rows):
    lines.extend([f"## {i}: {r['id']}",f"Source: {r['source']}",f"Model view: {r['model_view']}",f"Correct: {r['correct']['text']}",f"Shuffled: {r['shuffled']['text']}",''])
(HERE/'ALL_RECONSTRUCTIONS.md').write_text('\n\n'.join(lines),encoding='utf8')
(OUT/'checks.json').write_text(json.dumps(dict(status='passed',count=len(rows),unique_ids=len({r['id'] for r in rows}),cross_document_same_bin_derangement=True,shuffled_generation_equals_donor_reconstruction=True,protocol_and_driver_unchanged=True,checkpoint_data_and_tokenizer_identity=True,reserve_documents_disjoint=True,finite_scores=True),indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
