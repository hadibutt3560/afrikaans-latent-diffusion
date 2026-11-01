"""Installer, checksum/resume verification, refusal and unchanged-AE tests."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

root=Path(__file__).resolve().parents[1];patch=root/'hotfix-20261010'
sys.path.insert(0,str(root))
from common import read,write,sha

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

installer=module('repair_installer',patch/'install.py')
runner=module('repair_runner',patch/'run.py')
spec=read(patch/'patch.json')
with tempfile.TemporaryDirectory(dir=root/'local-checks') as temp:
    base=Path(temp);package=base/'package';out=base/'results';package.mkdir();out.mkdir()
    for name in spec['files']:shutil.copyfile(root/name,package/name)
    write(package/'bundle-manifest.json',{'files':{n:v['before'] for n,v in spec['files'].items()}})
    write(out/'stages.json',{'smoke':{'complete':True},'ae_train':{'complete':False,'deadline':time.time()+3600}})
    baseline=sha(package/'bundle-manifest.json')
    installer.install(package,out,patch)
    assert runner.verify_bundle(package)==baseline
    installer.install(package,out,patch)
    (package/'engine.py').write_text('damaged')
    try:runner.verify_bundle(package)
    except ValueError:pass
    else:raise AssertionError('Corruption not detected')
with tempfile.TemporaryDirectory(dir=root/'local-checks') as temp:
    base=Path(temp);package=base/'package';out=base/'results';package.mkdir();out.mkdir()
    write(package/'bundle-manifest.json',{'files':{}})
    write(out/'stages.json',{'smoke':{'complete':True},'benchmark':{'complete':False}})
    try:installer.install(package,out,patch)
    except AssertionError:pass
    else:raise AssertionError('Late install not refused')
old=ast.parse((root/'engine.py').read_text());new=ast.parse((patch/'engine.py').read_text())
def functions(tree):return {n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
before,after=functions(old),functions(new)
assert {n for n in before if before[n]!=after[n]}=={'loss','train'}
for n in ('ae_model','ae_forward','amp','validate','reconstruct'):assert before[n]==after[n]
print('PASS: install, idempotence, patched resume verifier, corruption rejection, late-install refusal, AE functions unchanged')
sys.modules['run']=runner
tempfile.tempdir=str(root/'local-checks')
protocol=module('patched_protocol_tests',root/'tests'/'test_protocol.py')
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(protocol))
assert result.wasSuccessful()
