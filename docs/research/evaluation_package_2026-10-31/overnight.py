"""Waits for the pinned downloads, records their hashes in PROTOCOL.md Appendix C, then runs the dry run and the full run."""
import ctypes, json, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=Path('E:/afrikaans-eval-20261031')
PY=str(E/'env/Scripts/python.exe')
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000|0x00000001)   # no idle sleep while this runs

def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)

while not (E/'pins.json').exists():
    if 'Traceback' in (E/'download.log').read_text(encoding='utf8',errors='replace'): log('download failed'); sys.exit(1)
    time.sleep(30)
log('downloads complete')

pins=json.loads((E/'pins.json').read_text(encoding='utf8'))
versions=subprocess.run([PY,'-c',"import importlib.metadata as m;print({k:m.version(k) for k in ('fasttext-turbo','spylls','mauve-text','faiss-cpu','sentence-transformers','matplotlib')})"],capture_output=True,text=True).stdout.strip()
lines=['','## Appendix C. Downloaded resources (SHA-256)','',
       'Recorded from `E:/afrikaans-eval-20261031/pins.json` after download, before any run. Package versions in the new environment: `'+versions+'`.','',
       '| Resource | File | SHA-256 |','|---|---|---|']
for key,g in pins.items():
    label=f"{g['repo']} @ {g['revision'][:12]}" if 'repo' in g else f"llama.cpp {g['tag']}" if 'tag' in g else key
    for f,h in g['files'].items(): lines.append(f'| {label} | `{f}` | `{h}` |')
proto=HERE/'PROTOCOL.md'; text=proto.read_text(encoding='utf8')
if '## Appendix C.' not in text: proto.write_text(text.rstrip('\n')+'\n'+'\n'.join(lines)+'\n',encoding='utf8')
log('appendix C written')

dry=subprocess.run([PY,str(HERE/'run_package.py'),'--dry-run'],capture_output=True,text=True,encoding='utf8',errors='replace')
log('dry run',dry.returncode,dry.stdout[-800:],dry.stderr[-1500:])
if dry.returncode: sys.exit(2)
with open(E/'run.log','w',encoding='utf8') as f:
    code=subprocess.run([PY,str(HERE/'run_package.py')],stdout=f,stderr=subprocess.STDOUT).returncode
log('run finished',code)
sys.exit(code)
