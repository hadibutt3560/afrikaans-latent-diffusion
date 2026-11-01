"""Create review artifacts on success, gate failure, or execution failure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}

def render(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=read(out/'summary.json');failure=read(out/'failure.json')
    baseline=read(out/'development-B0.json').get('metrics',{})
    records=[]
    if baseline:records.append(dict(condition='B0',selected_step=0,**baseline))
    for arm,values in summary.get('best',{}).items():
        records.append(dict(condition=arm,selected_step=values['step'],**values['metrics']))
    if records:
        with (out/'reconstruction-table.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Full decoder experiment: automatic results','',
           f"Status: {summary.get('status','not completed')}. Reason: {summary.get('reason','see logs')}.",'',
           '**Researcher review remains required. These results do not by themselves establish the proposal hypothesis.**','']
    if failure:lines.extend(['Execution error: `'+str(failure.get('error',failure))+'`.',''])
    lines.extend([f"Selected candidate: {summary.get('candidate') or 'none'}.",
                  f"Final update counts: {summary.get('final_steps',{})}.",
                  f"Optimizer/update-check time: {summary.get('training_seconds',0)/3600:.3f} hours.",
                  f"Total measured time: {summary.get('total_seconds',0)/3600:.3f} hours.",'',
                  '| Condition | Selected update | Development chrF | Exact | Numeric recall | Numeric precision |',
                  '|---|---:|---:|---:|---:|---:|'])
    for r in records:
        lines.append(f"| {r['condition']} | {r['selected_step']} | {r['chrf']:.3f} | {r['exact']}/{r['n']} | {r['numeric_recall']:.3f} | {r['numeric_precision']:.3f} |")
    lines.extend(['','Development scores are adaptively selected on 81 previously inspected passages. See selection.json for every check.',''])
    confirm=summary.get('confirmation')
    if isinstance(confirm,dict):
        lines.extend([f"Reserve automatic reconstruction gate: {confirm.get('automatic_reconstruction_gate_passed')}.",
            f"Paired-document chrF difference interval: {confirm.get('candidate_minus_baseline_chrf_ci')}.",
            'Full reserve metrics and review requirements are in summary.json and researcher-review.csv.',''])
    else:lines.extend([f'Reserve evaluation: {confirm or "not reached"}.',''])
    transfer=summary.get('transfer',{})
    if isinstance(transfer,dict) and transfer.get('status')=='completed':
        lines.extend(['| Decoder | Samples | mGPT PPL | Mean words | Distinct-2 | Empty | Train matches |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        table=[]
        for arm,r in transfer['metrics'].items():
            lines.append(f"| {arm} | {r['n']} | {r['ppl']} | {r['mean_words']:.2f} | {r['distinct_2']:.4f} | {r['empty']} | {r['train_matches']} |")
            table.append(dict(condition=arm,**{k:v for k,v in r.items() if k!='length_bands'}))
        with (out/'generation-table.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        lines.extend(['',f"Automatic generation guardrails: {transfer['automatic_transfer_guardrails_passed']}.",
            f"Paired PPL-ratio 95% interval: {transfer['intervals']['candidate_over_B0_ppl']}.",
            'See generation/summary.json for length bands, repetition, copying and uncertainty. Review the 64 blinded pairs and all flagged samples before interpreting quality.',''])
    else:lines.extend([f"Generation evaluation: {transfer or 'not reached'}.",''])
    lines.extend(['This is one training seed and one fixed diffusion checkpoint. No trained AR baseline is rerun here. External PPL is not a fidelity or linguistic-quality certificate. Unrelated real text is not used as a per-output reference. Historical test exposure remains disclosed; no new test examples were used.','',
                  'Keep this exploratory experiment separate from the historical all-seed benchmark. A failed gate is a reportable outcome, not a reason to select another checkpoint on the reserve.'])
    text='\n'.join(lines)+'\n';(out/'REPORT.md').write_text(text,encoding='utf-8')
    methods=('An exploratory decoder-adaptation follow-up compared adaptation of all twelve mT5 decoder blocks with a last-two-block control. Both conditions began from the original autoencoder and used the same training order and update counts at paired checkpoints. The encoder, Perceiver components, embeddings, output projection and diffusion model were frozen. Work was bounded by a measured wall-clock budget without quality-plateau stopping. Development selection used previously inspected validation passages; one candidate could proceed to disjoint reserve confirmation. The protocol scheduled conditional generation evaluation only after automatic reserve confirmation, using the same 256 sampled latents for the original and selected decoder with pinned mGPT scoring; the report below records whether this stage ran. This adaptively chosen follow-up does not replace the historical benchmark or constitute an independent confirmatory test.\n\n')
    (out/'MANUSCRIPT_INSERT.md').write_text('# Draft insertion requiring author review\n\n'+methods+text,encoding='utf-8')

def archive(out,package,job):
    render(out);manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='artifact-manifest.json':
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
            manifest.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=h.hexdigest(),
                                 included_in_review_archive=not(p.suffix=='.pt' and p.stat().st_size>20*1024**2)))
    (out/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target=out.parent/f'review-{out.name}.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=1) as tar:
        for row in manifest:
            if row['included_in_review_archive']:tar.add(out/row['path'],arcname=f"{out.name}/{row['path']}")
        tar.add(out/'artifact-manifest.json',arcname=f'{out.name}/artifact-manifest.json')
        for name in ('PLAN.md','bundle-manifest.sha256','train_full.py','transfer_eval.py','engine.py','export_report.py'):
            if (package/name).exists():tar.add(package/name,arcname=f'protocol/{name}')
        for suffix in ('out','err'):
            p=package/'logs'/f'af-full-12h-{job}.{suffix}'
            if p.exists():tar.add(p,arcname=f'logs/{p.name}')
    print('REVIEW_ARCHIVE='+str(target),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--job',default='local')
    a=p.parse_args();archive(a.output,Path(__file__).resolve().parent,a.job)
