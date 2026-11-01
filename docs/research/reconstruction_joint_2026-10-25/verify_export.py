"""Exercise new export with previously verified identity-scoring fixtures; not a fresh scorer run."""
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

REPO=Path(__file__).resolve().parents[3]
PACKAGE=REPO/'outputs/cluster-transfer/afrikaans-reconstruction-joint-20261025'
SOURCE=REPO/'outputs/decoder-dropout-verification-20261025-v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
spec=importlib.util.spec_from_file_location('report',PACKAGE/'export_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
baseline=read(SOURCE/'verification-baseline.json')
transfer=read(SOURCE/'generation/summary.json')
assert transfer['status']=='verification_only_no_training' and transfer['smoke_only']
assert transfer['n']==4 and set(transfer['metrics'])=={'B0','D0','D1'}
assert transfer['intervals']['candidate_over_B0_ppl']==[1.,1.]
assert transfer['candidate_over_control_intervals']['candidate_over_B0_ppl']==[1.,1.]
assert read(SOURCE/'verification.json')['optimizer_updates']==0
assert read(SOURCE/'gradient-verification.json')['optimizer_updates']==0
with tempfile.TemporaryDirectory(prefix='dropout-export-check-') as folder:
    out=Path(folder)/'job-smoke';out.mkdir()
    # Copy small review artifacts only; do not duplicate the identity model tensor.
    for p in SOURCE.rglob('*'):
        if p.is_file() and p.suffix!='.pt':
            target=out/p.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (out/'development-B0.json').write_text(json.dumps(baseline),encoding='utf-8')
    fixture=dict(transfer,status='completed')
    summary=dict(status='VERIFICATION_FIXTURE_NO_TRAINING',reason='identity test only',candidate='D1',
                 best={},final_steps={},training_seconds=0.,total_seconds=1.,
                 confirmation='Synthetic report fixture, not a scientific outcome',transfer=fixture)
    (out/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    report.archive(out,PACKAGE,'smoke')
    target=out.parent/'review-job-smoke.tar.gz'
    with tarfile.open(target) as archive:
        names=set(archive.getnames())
        for name in ('REPORT.md','MANUSCRIPT_INSERT.md','reconstruction-table.csv','generation-table.csv','artifact-manifest.json',
                     'generation/researcher-generation-review.csv','generation/control-comparison/researcher-generation-review.csv'):
            assert f'job-smoke/{name}' in names,name
        assert 'protocol/train_joint.py' in names and 'protocol/joint.slurm' in names
        for row in read(out/'artifact-manifest.json'):
            if row['included_in_review_archive']:
                import hashlib
                actual=hashlib.sha256(archive.extractfile('job-smoke/'+row['path']).read()).hexdigest()
                assert actual==row['sha256']
    assert target.with_name(target.name+'.sha256').read_text().split()[0]==report.digest(target)
print('Three-way identity scoring, positive report, both review sheets and every exported artifact hash passed.')
