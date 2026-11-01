"""Export positive, negative, inadmissible and catchable-failure results in this job."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tarfile

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def table(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def render(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    s=read(out/'summary.json');failure=read(out/'failure.json');exit_record=read(out/'job-exit.json')
    baseline=read(out/'development-B0.json').get('metrics',{})
    records=[dict(condition='B0',step=0,**baseline)] if baseline else []
    for arm,best in s.get('best',{}).items():records.append(dict(condition=arm,step=best['step'],**best['metrics']))
    table(out/'reconstruction-table.csv',records)
    trajectory=[]
    for arm,history in read(out/'selection.json').get('history',{}).items():
        trajectory.extend(dict(condition=arm,step=r['step'],**r['metrics']) for r in history)
    table(out/'learning-curves.csv',trajectory)
    lines=['# Joint reconstruction adaptation: automatic results','',
        'D0: full decoder only. D1: full decoder plus reconstruction mapping. Both start from B0; dropout disabled in both.','',
        f"Status: {s.get('status','not completed')}; reason: {s.get('reason','see failure/logs')}.",
        f"Driver exit: {exit_record.get('exit_code','not yet recorded')}.",
        f"Selected candidate: {s.get('candidate') or 'none'}.",
        f"Final update counts: {s.get('final_steps',{})}.",
        f"Update/replay time: {s.get('training_seconds',0)/3600:.3f} h; driver time: {s.get('total_seconds',0)/3600:.3f} h.",'',
        '**Researcher inspection remains required. Automatic checks alone do not establish linguistic quality or the proposal hypothesis.**','']
    if failure:lines+=['Execution failure: `'+str(failure)+'`.','']
    if records:
        lines+=['| Condition | Selected step | chrF | Exact | Numeric recall | Numeric precision |',
                '|---|---:|---:|---:|---:|---:|']
        for r in records:lines.append(f"| {r['condition']} | {r['step']} | {r['chrf']:.4f} | {r['exact']}/{r['n']} | {r['numeric_recall']:.4f} | {r['numeric_precision']:.4f} |")
    lines+=['','Development selection repeatedly uses the same 81 exposed passages. All checks are in learning-curves.csv and selection.json; do not report selected gains as independent confirmation.','']
    c=s.get('confirmation')
    if isinstance(c,dict):
        lines += [f"Reserve reconstruction gate versus B0: {c.get('automatic_reconstruction_gate_passed')}.",
            f"D1 minus B0 chrF 95% document-bootstrap interval: {c.get('candidate_minus_baseline_chrf_ci')}.",
            f"Joint-adaptation-specific reserve effect supported: {c.get('joint_effect_supported')}.",
            f"D1 minus D0 chrF interval: {c.get('joint_minus_control_chrf_ci')}.",'']
        table(out/'reserve-table.csv',[dict(condition=a,**m) for a,m in c.get('metrics',{}).items()])
    else:lines += [f"Reserve: {c or 'not reached'}.",'']
    t=s.get('transfer',{})
    if t.get('status')=='completed':
        table(out/'generation-table.csv',[dict(condition=a,**{k:v for k,v in m.items() if k!='length_bands'}) for a,m in t['metrics'].items()])
        lines+=['| Decoder | n | mGPT PPL | Mean words | Distinct2 | Empty | Train matches |','|---|---:|---:|---:|---:|---:|---:|']
        for a,m in t['metrics'].items():lines.append(f"| {a} | {m['n']} | {m['ppl']} | {m['mean_words']:.2f} | {m['distinct_2']:.4f} | {m['empty']} | {m['train_matches']} |")
        lines+=['',f"Generation non-regression guards vs B0: {t['automatic_transfer_guardrails_passed']}; vs D0: {t.get('control_guardrails_passed')}.",
            f"D1/B0 PPL-ratio 95% interval: {t['intervals']['candidate_over_B0_ppl']}.",
            f"D1/D0 paired uncertainty: {t.get('candidate_over_control_intervals')}.",
            'See generation/summary.json for length bands and repetition. Review both sets of 64 blinded pairs and all flags. These checks do not demonstrate better language quality by themselves.','']
    else:lines += [f"Generation: {t or 'not reached'}.",'']
    lines += ['The reserve is disjoint from current development but belongs to historically exposed validation data. No new test tuning occurred. The historical benchmark and qualified AR comparison remain unchanged.',
              'This is one training seed with one fixed denoiser. Document/sample bootstrap intervals do not measure training-seed variability. Unrelated real text is not a per-output reference. Researcher review and any AI-assisted interpretation must be distinguished.','',
              'A failed gate is a completed negative result. Do not consume the reserve with another checkpoint, lower thresholds or automatically launch another run.']
    content='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(content,encoding='utf-8')
    method=('An adaptively chosen Afrikaans follow-up compared full-decoder adaptation against joint adaptation of the reconstruction mapping and full decoder. Both began from the original AE checkpoint with identical optimizer, data order, cached latents, disabled dropout and decoding. Encoder and compression network stayed frozen, preserving the latent space and diffusion checkpoints. Maximum 26,620 updates per arm; paired plateau stopping after at least 12,000 updates; 12-hour total budget. Selection used repeatedly exposed development data; reserve and generation evaluation were conditional on fixed gates. This is literature-motivated, not a replication of LD4LG whole-autoencoder training. One seed cannot establish training-seed uncertainty.\n\n')
    (out/'MANUSCRIPT_INSERT.md').write_text('# Draft insertion for author review\n\n'+method+content,encoding='utf-8')

def archive(out,package,job,exit_code=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'job-exit.json').write_text(json.dumps(dict(exit_code=exit_code,utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    render(out)
    manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='artifact-manifest.json':
            manifest.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=digest(p),
                                 included_in_review_archive=not(p.suffix=='.pt' and p.stat().st_size>20*1024**2)))
    (out/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target=out.parent/f'review-{out.name}.tar.gz';tmp=target.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for r in manifest:
            if r['included_in_review_archive']:tar.add(out/r['path'],arcname=f"{out.name}/{r['path']}")
        tar.add(out/'artifact-manifest.json',arcname=f'{out.name}/artifact-manifest.json')
        for p in sorted(package.iterdir()):
            if p.is_file() and (p.suffix in ('.py','.sh','.slurm') or p.name in ('PLAN.md','bundle-manifest.sha256','helper-provenance.json')):
                tar.add(p,arcname=f'protocol/{p.name}')
        for suffix in ('out','err'):
            p=package/'logs'/f'af-joint-final-{job}.{suffix}'
            if p.exists():tar.add(p,arcname=f'logs/{p.name}')
        scheduler=package/'outputs'/f'scheduler-{job}.txt'
        if scheduler.exists():tar.add(scheduler,arcname=f'logs/{scheduler.name}')
    tmp.replace(target)
    target.with_name(target.name+'.sha256').write_text(f'{digest(target)}  {target.name}\n')
    print('REVIEW_ARCHIVE='+str(target),flush=True)
    print('REVIEW_SHA256='+digest(target),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--job',default='local')
    p.add_argument('--exit-code',type=int,default=0);a=p.parse_args()
    archive(a.output,Path(__file__).resolve().parent,a.job,a.exit_code)
