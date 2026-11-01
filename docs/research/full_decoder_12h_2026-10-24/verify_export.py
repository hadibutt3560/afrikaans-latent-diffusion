"""Exercise positive report/archive export using inference-only verification data."""
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'outputs/cluster-transfer/afrikaans-full-decoder-allinone-20261024'
SOURCE=ROOT/'outputs/full-decoder-allinone-verification-20261024'
spec=importlib.util.spec_from_file_location('report',PACKAGE/'export_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
with tempfile.TemporaryDirectory(prefix='decoder-export-check-') as directory:
    out=Path(directory)/'job-smoke'
    shutil.copytree(SOURCE,out)
    baseline=json.loads((out/'verification-baseline.json').read_text(encoding='utf-8'))
    (out/'development-B0.json').write_text(json.dumps(baseline),encoding='utf-8')
    transfer=json.loads((out/'generation/summary.json').read_text(encoding='utf-8'))
    summary=dict(status='VERIFICATION_ONLY_NO_TRAINING',candidate='SMOKE_IDENTITY',
        best={},final_steps={},training_seconds=0.,total_seconds=1.,
        confirmation='Synthetic report fixture; not a scientific outcome',transfer=transfer)
    (out/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    report.archive(out,PACKAGE,'smoke')
    with tarfile.open(out.parent/'review-job-smoke.tar.gz') as archive:
        names=set(archive.getnames())
        for name in ('REPORT.md','MANUSCRIPT_INSERT.md','reconstruction-table.csv','generation-table.csv','artifact-manifest.json'):
            assert f'job-smoke/{name}' in names,name
        assert 'protocol/train_full.py' in names
        assert 'job-smoke/generation/researcher-generation-review.csv' in names
    assert 'SMOKE_IDENTITY' in (out/'REPORT.md').read_text(encoding='utf-8')
    print('Positive report/archive export passed; fixture removed; no research outputs modified.')
