import os,sys,json,hashlib,importlib.util,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
OUT=ROOT/'outputs/stage4-final-evaluation-20261023'
BUNDLE=ROOT/'outputs/longrun-local-inputs'; EVAL=ROOT/'outputs/longrun-local-evaluation'
REV='40897bd7c8b47a76802c411108ca6220438b8b40'
os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.stdout.reconfigure(encoding='utf8')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import torch,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from huggingface_hub import snapshot_download
 torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 OUT.mkdir(exist_ok=True)
 if any(OUT.iterdir()): raise FileExistsError('Refusing to overwrite scoring output')
 path=BUNDLE/'tools/compare_afrikaans_generators.py'
 assert sha(path)==read(EVAL/'local-contract.json')['inputs']['code']['tools/compare_afrikaans_generators.py']
 spec=importlib.util.spec_from_file_location('frozen',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 snap=snapshot_download('ai-forever/mGPT',revision=REV,local_files_only=True)
 tok=AutoTokenizer.from_pretrained(snap,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,dtype='float32').cuda().eval().requires_grad_(False)
 anchors=[]
 with torch.inference_mode():
  for name in ('small-seed42','published-seed42'):
   for row in read(EVAL/'runs'/name/'validation-mgpt.json')['rows'][:2]:
    score=mod.score_text(model,tok,row['text'],'cuda'); old=row['mgpt']
    assert score['tokens']==old['tokens'] and abs(score['nll']-old['nll'])/max(1,abs(old['nll']))<1e-4
    anchors.append(dict(run=name,seed=row['seed'],old=old,new=score))
  source=ROOT/'outputs/reconstruction-audit-20261022/examples.jsonl'
  rows=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
  start=time.monotonic()
  with (OUT/'reconstruction-scores.jsonl').open('x',encoding='utf8') as f:
   for r in rows:
    value=dict(id=r['id'],document_id=r['document_id'],source=r['model_view'],reconstruction=r['correct']['text'],source_mgpt=mod.score_text(model,tok,r['model_view'],'cuda'),reconstruction_mgpt=mod.score_text(model,tok,r['correct']['text'],'cuda'))
    f.write(json.dumps(value,ensure_ascii=False)+'\n')
  contract=dict(revision=REV,scorer_sha256=sha(path),input_sha256=sha(source),protocol_sha256=sha(HERE/'PROTOCOL.md'),script_sha256=sha(Path(__file__)),precision='FP32',context=512,stride=256,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),anchors=anchors,count=len(rows),seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
  (OUT/'scoring-contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
 print('Completed81 reconstruction/source pairs; four scorer anchors passed.')
if __name__=='__main__': main()
