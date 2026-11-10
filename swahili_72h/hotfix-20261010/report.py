"""Report complete or failed runs, blinded human sheets, plots, portable archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import tarfile
from common import read,rows,write,sha
from metrics import diversity,aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--no-archive',action='store_true');a=p.parse_args()
    out=a.out;groups={};review=[];key=[]
    for path in sorted((out/'scores').glob('*.json')):
        if '.partial.' in path.name:continue
        data=read(path);groups[path.stem]=dict(**diversity([x['text'] for x in data]),external_lm=aggregate([x['score'] for x in data]))
        sample=random.Random(19273).sample(data,min(50,len(data)))
        for i,row in enumerate(sample):
            rid=hashlib.sha256(f'{path.stem}:{i}:928127'.encode()).hexdigest()[:16]
            review.append(dict(review_id=rid,text=row['text'],grammar_1_to_5='',coherence_1_to_5='',natural_swahili_1_to_5='',comments=''))
            key.append(dict(review_id=rid,group=path.stem,seed=row.get('seed'),source_id=row.get('id')))
    random.Random(719).shuffle(review)
    for name,data,fields in [('blind-review.csv',review,['review_id','text','grammar_1_to_5','coherence_1_to_5','natural_swahili_1_to_5','comments']),
                             ('review-key.csv',key,['review_id','group','seed','source_id'])]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    completion=read(out/'completion.json') if (out/'completion.json').exists() else {'status':'partial'}
    training_status=[read(p) for p in [out/'ae/status.json',*list((out/'runs').glob('*/status.json'))] if p.exists()]
    results=dict(completion=completion,groups=groups,productive_optimization_hours=sum(s.get('optimization_seconds',0) for s in training_status)/3600,
        validation_hours=sum(s.get('validation_seconds',0) for s in training_status)/3600,
        human_review='Not yet performed; distribute blind-review.csv without review-key.csv.',
        limitations=['File-level supplied splits retained despite documented overlap; no original document boundaries.',
        '48-word corpus windows may span original documents; coherence conclusions are limited.',
        'mGPT perplexity is exploratory, may have pretraining overlap, and does not establish fluency.',
        'Perplexities from different tokenizers or languages are not directly comparable.',
        'Timed budgets may yield unequal steps; inspect run duration and convergence before architecture claims.',
        'AE has one initialization seed; diffusion seeds quantify diffusion variability only.'])
    if (out/'runtime-patch.json').exists():
        results['runtime_patch']=read(out/'runtime-patch.json')
        results['diffusion_learning_check']=read(out/'diffusion-learning-check.json') if (out/'diffusion-learning-check.json').exists() else None
        results['limitations'].append('Initial diffusion resume smoke did not verify learning. A PyTorch 2.7.1 autocast-cache repair was installed before diffusion benchmarking/training; see runtime-patch.json and diffusion-learning-check.json for provenance and replacement checks.')
    if (out/'selection.json').exists():
        results['selection']=read(out/'selection.json');results['runs']=[];family={}
        for selected in results['selection']['runs']:
            name=selected['run'];folder=out/'runs'/name
            item=dict(run=name,validation=selected['validation'],test=read(folder/'test.json') if (folder/'test.json').exists() else None,
                training=read(folder/'status.json') if (folder/'status.json').exists() else None)
            results['runs'].append(item);family.setdefault(name.rsplit('-',1)[0],[]).append(item)
        results['family_summary']={}
        for name,items in family.items():
            vals=[x['test']['loss'] for x in items if x['test']]
            results['family_summary'][name]=dict(completed_test_seeds=len(vals),mean_test_mse=statistics.mean(vals) if vals else None,
                sample_sd_test_mse=statistics.stdev(vals) if len(vals)>1 else None)
    write(out/'results.json',results)
    text=['# Swahili experiment results','',f"Status: **{completion['status']}**",'',results['human_review'],'',*['- '+x for x in results['limitations']],'']
    if completion.get('error'):text+=['Failure: '+completion['error'],'']
    text+=['| Group | Samples | Empty | Token-weighted mGPT PPL |','|---|---:|---:|---:|']
    for name,g in groups.items():text.append(f"| {name} | {g['samples']} | {g['empty']} | {g['external_lm']['token_weighted_perplexity']} |")
    if results.get('runs'):
        text+=['','| Run | Validation MSE | Test MSE | Optimizer steps | Training-stage seconds |','|---|---:|---:|---:|---:|']
        for r in results['runs']:
            t=r['training'] or {};text.append(f"| {r['run']} | {r['validation']['loss']} | {(r['test'] or {}).get('loss')} | {t.get('step')} | {t.get('seconds')} |")
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    # Plot generation uses the fixed local matplotlib dependency, no remote metrics.
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['pdf.use14corefonts']=True
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for path in sorted((out/'runs').glob('*/metrics.jsonl')):
            r=[x for x in rows(path) if x['kind']=='validation'];axes[0].plot([x['step'] for x in r],[x['loss'] for x in r],label=path.parent.name)
        axes[0].set(xlabel='Optimizer steps',ylabel='Fixed validation MSE',title='Validation checkpoint trajectory')
        if axes[0].lines:axes[0].legend()
        labels=[k for k,v in groups.items() if v['external_lm']['token_weighted_perplexity'] is not None]
        axes[1].bar(labels,[groups[k]['external_lm']['token_weighted_perplexity'] for k in labels]);axes[1].tick_params(axis='x',rotation=80)
        axes[1].set(ylabel='Token-weighted mGPT perplexity',title='Exploratory external scoring')
        fig.tight_layout();fig.savefig(out/'results.png',dpi=180);fig.savefig(out/'results.pdf');plt.close(fig)
        if (out/'plot-failure.json').exists():(out/'plot-failure.json').unlink()
    except Exception as exc:
        write(out/'plot-failure.json',dict(error=repr(exc)))
        if completion['status']=='complete':
            completion.update(status='partial',error='Plot generation failed');write(out/'completion.json',completion)
            results['completion']=completion;write(out/'results.json',results)
            (out/'RESULTS.md').write_text('\n'.join(text).replace('Status: **complete**','Status: **partial**')+'\n\nPlot generation failed; see plot-failure.json.\n',encoding='utf8')
    # Include reproducible source/config/data manifest; large raw inputs remain in transfer bundle.
    provenance=out/'provenance';provenance.mkdir(exist_ok=True)
    import shutil
    for name in ('config.json','bundle-manifest.json','source-provenance.json','data-decision.json','RUNBOOK.md'):
        if (a.root/name).exists():shutil.copyfile(a.root/name,provenance/name)
    for name in ('audit.json','split_manifest.json'):shutil.copyfile(a.root/'data'/name,provenance/name)
    if not a.no_archive:
        paths=[p for p in out.rglob('*') if p.is_file() and not any(x in ('cache','smoke','benchmark') for x in p.relative_to(out).parts)
               and p.name not in ('results.tar','results.tar.tmp','results.tar.sha256','results-manifest.json') and not p.name.endswith('.tmp')]
        write(out/'results-manifest.json',{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in paths})
        archive=out/'results.tar';temp=out/'results.tar.tmp'
        with tarfile.open(temp,'w') as tar:
            for p in paths+[out/'results-manifest.json']:tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
        temp.replace(archive);(out/'results.tar.sha256').write_text(sha(archive)+'  results.tar\n',encoding='ascii')
    print('Results:',out,flush=True)


if __name__=='__main__':main()
