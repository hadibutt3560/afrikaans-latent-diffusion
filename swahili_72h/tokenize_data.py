"""Offline tokenizer audit and fixed-length model views; canonical text is untouched."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import T5Tokenizer
from common import rows, read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent);a=p.parse_args()
    root=a.root; data=root/'data'; out=root/'tokens';out.mkdir(exist_ok=True)
    tokenizer=T5Tokenizer.from_pretrained(str(root/'assets/mt5-base'),local_files_only=True,legacy=True)
    audit=read(data/'audit.json'); report={}
    for split,n in audit['record_counts'].items():
        dest=out/f'{split}.npy'; array=np.lib.format.open_memmap(str(dest)+'.tmp',mode='w+',dtype='int32',shape=(n,192))
        hist=Counter();unknown=total=truncated=0
        with (out/f'{split}-views.jsonl').open('w',encoding='utf8') as f:
            import json
            for i,row in enumerate(rows(data/f'{split}.jsonl')):
                ids=tokenizer.encode(row['text'],add_special_tokens=True);hist[len(ids)]+=1
                unknown+=ids.count(tokenizer.unk_token_id);total+=len(ids);truncated+=len(ids)>192
                model_ids=(ids[:191]+[tokenizer.eos_token_id]) if len(ids)>192 else ids
                array[i]=model_ids+[tokenizer.pad_token_id]*(192-len(model_ids))
                f.write(json.dumps(dict(id=row['id'],row=i,original_tokens=len(ids),truncated=len(ids)>192))+'\n')
                if i and i%100000==0:print(split,i,flush=True)
        array.flush();del array;Path(str(dest)+'.tmp').replace(dest)
        report[split]=dict(records=n,tokens=total,unknown_tokens=unknown,unknown_fraction=unknown/max(total,1),
            truncated=truncated,truncated_fraction=truncated/n,length_histogram=dict(hist),sha256=sha(dest))
    write(out/'audit.json',dict(max_length=192,policy='T5Tokenizer legacy=True; add EOS; right truncate only the model view to 191 tokens+EOS; raw canonical text unchanged; pad id 0.',splits=report))
    print({s:{k:v for k,v in r.items() if k!='length_histogram'} for s,r in report.items()})


if __name__=='__main__':main()
