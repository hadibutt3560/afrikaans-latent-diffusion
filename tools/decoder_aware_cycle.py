"""Controlled frozen-AE diffusion fine-tuning and unconditional evaluation."""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tools.trace_afrikaans_inference import DiffusionTransformer, GaussianDiffusion, get_latent_model
from tools.compare_afrikaans_generators import score_text, diversity

REVISION='40897bd7c8b47a76802c411108ca6220438b8b40'
STOP=False


def stop_signal(*args):
    global STOP
    STOP=True


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,data):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def auxiliary_objective(velocity_loss,decoder_loss,weight):
    return velocity_loss+weight*decoder_loss


def make_diffusion(d,a):
    net=DiffusionTransformer(tx_dim=d.tx_dim,tx_depth=d.tx_depth,heads=d.tx_dim//64,
        latent_dim=a.dim_ae,max_seq_len=d.max_seq_len,self_condition=d.self_condition,
        scale_shift=d.scale_shift,dropout=0 if d.disable_dropout else .1,
        class_conditional=False,num_classes=0,class_unconditional_prob=d.class_unconditional_prob,
        seq2seq=False,seq2seq_context_dim=768,num_dense_connections=d.num_dense_connections)
    obj=GaussianDiffusion(net,max_seq_len=32,sampling_timesteps=50,sampler='ddim',
        train_schedule=d.train_schedule,sampling_schedule=d.sampling_schedule,loss_type=d.loss_type,
        objective=d.objective,train_prob_self_cond=d.train_prob_self_cond,
        seq2seq_unconditional_prob=d.seq2seq_unconditional_prob,scale=d.scale)
    obj.using_latent_model=True
    obj.l2_normalize=a.l2_normalize_latents
    return obj.cuda()


@torch.no_grad()
def encode_cache(ae,tokenizer,path,count,seed):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines()]
    random.Random(seed).shuffle(rows)
    values=[]
    for i,row in enumerate(rows[:count]):
        batch=tokenizer(row['text'],return_tensors='pt',padding='max_length',truncation=True,max_length=128).to('cuda')
        latent=ae.get_diffusion_latent(ae.get_encoder()(**batch),batch.attention_mask).float().cpu()
        labels=batch.input_ids.cpu()
        labels[batch.attention_mask.cpu()==0]=-100
        values.append((latent,labels,row['id']))
        if (i+1)%32==0:
            print(f'Encoded {path}: {i+1}/{min(count,len(rows))}',flush=True)
    return values


def save_checkpoint(path,model,ema,opt,step,arm,protocol):
    state=dict(model=model.state_dict(),ema={'ema_model.'+k:v for k,v in ema.state_dict().items()},
        opt=opt.state_dict(),step=step,using_latent_model=True,arm=arm,protocol=protocol)
    temp=path.with_suffix('.tmp')
    torch.save(state,temp)
    temp.replace(path)


@torch.no_grad()
def generate(model,ae,tokenizer,seeds):
    model.eval()
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        latent,_=model.sample(1,torch.tensor([32],device='cuda'))
        if not torch.isfinite(latent).all():
            raise FloatingPointError('Non-finite generated latent')
        ids=ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(latent)),
            max_length=128,min_length=5,num_beams=4,do_sample=False,
            no_repeat_ngram_size=3,repetition_penalty=1.2)
        tokens=ids[0]
        eos=(tokens==tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        n=int(eos[0])+1 if len(eos) else len(tokens)
        rows.append(dict(seed=seed,text=tokenizer.decode(tokens[:n],skip_special_tokens=True).strip(),
            decoder_tokens=n,ended_with_eos=bool(len(eos)),reached_length_cap=n>=128))
        print(f'Generated seed {seed}: {rows[-1]["text"]}',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diffusion-dir',type=Path,default=Path('outputs/cluster-results/diffusion-50k-retry-20260920'))
    p.add_argument('--autoencoder-dir',type=Path,default=Path('outputs/cluster-results/afrikaans-56531/best'))
    p.add_argument('--dataset-dir',type=Path,default=Path('data/nchlt_afrikaans/processed/stage_5_7'))
    p.add_argument('--model-cache',default='google/mt5-base')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--arm-seconds',type=float,default=240)
    p.add_argument('--train-examples',type=int,default=128)
    p.add_argument('--valid-examples',type=int,default=16)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--weight',type=float,default=.02)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--sample-count',type=int,default=8)
    p.add_argument('--sample-start',type=int,default=142)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--score-only',action='store_true')
    args=p.parse_args()
    if args.score_only:
        score_results(args.output_dir)
        return
    if min(args.steps,args.accumulation,args.save_every,args.train_examples,args.valid_examples,args.sample_count)<1 or args.arm_seconds<=0:
        p.error('Positive counts and budget required')
    if args.weight<=0:
        p.error('Treatment weight must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Usable CUDA GPU required')
    torch.ones(16,device='cuda').square().sum().item()
    d=SimpleNamespace(**json.loads((args.diffusion_dir/'args.json').read_text()))
    a=SimpleNamespace(**json.loads((args.autoencoder_dir/'args.json').read_text()))
    if d.objective!='pred_v' or d.normalize_latent or d.class_conditional or (a.dim_ae,a.num_encoder_latents)!=(64,32):
        raise ValueError('Unsupported checkpoint protocol')
    protocol=dict(intervention='velocity_plus_frozen_decoder_ce',weight=args.weight,lr=args.lr,
        ema_decay=.99,initialization='source EMA weights; new matched AdamW optimizers, not historical optimizer resume',
        source_diffusion_sha256=digest(args.diffusion_dir/'model.pt'),
        source_ae_sha256=digest(args.autoencoder_dir/'model.pt'),
        data_hashes={s:digest(args.dataset_dir/f'{s}.jsonl') for s in ('train','valid')},
        train_examples=args.train_examples,valid_examples=args.valid_examples,accumulation=args.accumulation,
        selection_seed=24680,training_seed=24680,sampling_seeds=list(range(args.sample_start,args.sample_start+args.sample_count)),
        sampling=dict(sampler='ddim',steps=50,max_length=128,min_length=5,num_beams=4,
            no_repeat_ngram_size=3,repetition_penalty=1.2),precision='float32',
        success='Blinded fluent-human mean coherence and grammar each >=0.5/5 better than continued control; >=60% paired preference, no material collapse. AI/automatic scores exploratory only.',
        scope='Exploratory validation; no test evaluation',torch=torch.__version__,gpu=torch.cuda.get_device_name())
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError('Use new output or explicit --resume')
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    if args.resume:
        old=json.loads((args.output_dir/'protocol.json').read_text())
        if old!=protocol:
            raise ValueError('Resume protocol differs: source, hardware or settings changed')
    else:
        write(args.output_dir/'protocol.json',protocol)
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    model=make_diffusion(d,a)
    ckpt=torch.load(str(args.diffusion_dir/'model.pt'),map_location='cpu',mmap=True,weights_only=False)
    source={k[len('ema_model.'):]:v.clone() for k,v in ckpt['ema'].items() if k.startswith('ema_model.')}
    if not ckpt.get('using_latent_model') or ckpt['step']!=50000:
        raise ValueError('Expected active 50k latent checkpoint')
    model.load_state_dict(source,strict=True)
    del ckpt
    a.enc_dec_model=args.model_cache
    ae,tokenizer,_=get_latent_model(a)
    ckpt=torch.load(str(args.autoencoder_dir/'model.pt'),map_location='cpu',mmap=True,weights_only=False)
    ae.load_state_dict(ckpt['model'],strict=True)
    del ckpt
    ae.cuda().eval().requires_grad_(False)
    write(args.output_dir/'preflight.json',dict(status='passed',source_step=50000,ae_step=30500,
        gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),strict_load=True,
        decoder_start=ae.config.decoder_start_token_id,eos=ae.config.eos_token_id))
    if args.preflight:
        return
    train=encode_cache(ae,tokenizer,args.dataset_dir/'train.jsonl',args.train_examples,24680)
    valid=encode_cache(ae,tokenizer,args.dataset_dir/'valid.jsonl',args.valid_examples,24680)
    seeds=protocol['sampling_seeds']
    groups={}
    frozen_path=args.output_dir/'frozen-samples.json'
    if frozen_path.exists():
        groups['frozen']=json.loads(frozen_path.read_text())
    else:
        groups['frozen']=generate(model,ae,tokenizer,seeds)
        write(frozen_path,groups['frozen'])
    stats={}
    matched_limit=args.steps
    for arm,weight in (('control',0.),('decoder_ce',args.weight)):
        folder=args.output_dir/arm
        folder.mkdir(exist_ok=True)
        model.load_state_dict(source,strict=True)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=d.adam_weight_decay)
        step=0
        checkpoint_path=folder/'model.pt'
        if args.resume and checkpoint_path.exists():
            saved=torch.load(str(checkpoint_path),map_location='cuda',weights_only=False)
            if saved['protocol']!=protocol or saved['arm']!=arm:
                raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(saved['model'],strict=True)
            ema.load_state_dict({k[10:]:v for k,v in saved['ema'].items()},strict=True)
            opt.load_state_dict(saved['opt'])
            step=saved['step']
            del saved
        initial=step
        start=time.monotonic()
        save_roundtrip=False
        model.train()
        while step<matched_limit and time.monotonic()-start<args.arm_seconds and not STOP:
            opt.zero_grad(set_to_none=True)
            logged_v=logged_ce=0.
            for micro in range(args.accumulation):
                seed=24680+step*args.accumulation+micro
                torch.manual_seed(seed)
                random.seed(seed)
                idx=random.randrange(len(train))
                latent,labels,_=train[idx]
                latent,labels=latent.cuda(),labels.cuda()
                mask=torch.ones(latent.shape[:2],dtype=torch.bool,device='cuda')
                vloss,pred=model(latent,mask,class_id=None,return_x_start=True)
                ce=torch.zeros((),device='cuda')
                if weight:
                    encoded=BaseModelOutput(last_hidden_state=ae.get_decoder_input(pred))
                    ce=ae(encoder_outputs=encoded,labels=labels).loss
                loss=auxiliary_objective(vloss,ce,weight)/args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss; refusing optimizer update')
                loss.backward()
                logged_v+=float(vloss.detach())/args.accumulation
                logged_ce+=float(ce.detach())/args.accumulation
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if not all(torch.isfinite(v).all() for v in model.parameters()):
                raise FloatingPointError('Non-finite parameter; last saved checkpoint preserved')
            with torch.no_grad():
                for dest,src in zip(ema.parameters(),model.parameters()):
                    dest.lerp_(src,.01)
            step+=1
            row=dict(step=step,velocity_loss=logged_v,decoder_ce=logged_ce,grad_norm=float(norm),
                elapsed_seconds=time.monotonic()-start,auxiliary_weight=weight)
            with (folder/'training.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            if step%args.save_every==0:
                save_checkpoint(checkpoint_path,model,ema,opt,step,arm,protocol)
                # Exercise a real model/EMA/optimizer serialization round-trip once.
                if not save_roundtrip:
                    saved=torch.load(str(checkpoint_path),map_location='cuda',weights_only=False)
                    if saved['step']!=step or not all(torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items()):
                        raise RuntimeError('Checkpoint round-trip mismatch')
                    model.load_state_dict(saved['model'],strict=True)
                    ema.load_state_dict({k[10:]:v for k,v in saved['ema'].items()},strict=True)
                    opt.load_state_dict(saved['opt'])
                    del saved
                    save_roundtrip=True
                print(f'{arm}: step {step}/{matched_limit}, v={logged_v:.4f}, ce={logged_ce:.4f}',flush=True)
        save_checkpoint(checkpoint_path,model,ema,opt,step,arm,protocol)
        write(folder/'args.json',vars(d))
        write(folder/'latent_model_provenance.json',json.loads((args.diffusion_dir/'latent_model_provenance.json').read_text()))
        if arm=='control':
            matched_limit=step
        stats[arm]=dict(step=step,new_steps=step-initial,resumed_from_step=initial,
            training_seconds=time.monotonic()-start,
            checkpoint_roundtrip=save_roundtrip,stopped=STOP or step<args.steps)
        if STOP:
            write(args.output_dir/'summary.json',dict(status='interrupted_resumable',training=stats))
            return
        ema.eval()
        val=[]
        with torch.no_grad():
            for i,(latent,labels,_) in enumerate(valid):
                torch.manual_seed(34680+i)
                random.seed(34680+i)
                latent=latent.cuda()
                val.append(float(ema(latent,torch.ones(latent.shape[:2],dtype=torch.bool,device='cuda'),class_id=None)))
        stats[arm]['validation_velocity_loss']=statistics.mean(val)
        groups[arm]=generate(ema,ae,tokenizer,seeds)
        write(args.output_dir/f'{arm}-samples.json',groups[arm])
        del ema,opt
        gc.collect()
        torch.cuda.empty_cache()
    status='completed_matched' if stats['control']['step']==stats['decoder_ce']['step'] else 'incomplete_unmatched'
    write(args.output_dir/'summary.json',dict(status=status,training=stats,success_demonstrated=False,
        human_review='pending',decision='Do not promote without human coherence/grammar confirmation'))
    blind=[]
    for arm,rows in groups.items():
        blind.extend(dict(arm=arm,**r) for r in rows)
    random.Random(44680).shuffle(blind)
    with (args.output_dir/'blind_review.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['id','text','grammar_1_to_5','coherence_1_to_5','notes'])
        for i,row in enumerate(blind,1):
            writer.writerow([i,row['text'],'','',''])
    with (args.output_dir/'review_key.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['id','arm','seed'])
        for i,row in enumerate(blind,1):
            writer.writerow([i,row['arm'],row['seed']])
    del model,ae,source,train,valid
    gc.collect()
    torch.cuda.empty_cache()
    score_results(args.output_dir)


def score_results(out):
    tok=AutoTokenizer.from_pretrained('ai-forever/mGPT',revision=REVISION,local_files_only=True)
    scorer=AutoModelForCausalLM.from_pretrained('ai-forever/mGPT',revision=REVISION,
        local_files_only=True,torch_dtype=torch.float32,low_cpu_mem_usage=True).cuda().eval().requires_grad_(False)
    report=dict(scorer='ai-forever/mGPT',revision=REVISION,context=512,stride=256,groups={},human_review='pending')
    for arm in ('frozen','control','decoder_ce'):
        rows=json.loads((out/f'{arm}-samples.json').read_text())
        scores=[score_text(scorer,tok,r['text'],'cuda') for r in rows]
        if any(s is None for s in scores):
            raise ValueError('Empty generation; no silently omitted sample')
        report['groups'][arm]=dict(diversity([r['text'] for r in rows]),
            token_weighted_perplexity=math.exp(sum(s['nll'] for s in scores)/sum(s['tokens'] for s in scores)),
            median_perplexity=statistics.median(s['perplexity'] for s in scores),
            mean_decoder_tokens=statistics.mean(r['decoder_tokens'] for r in rows),
            eos=sum(r['ended_with_eos'] for r in rows),length_cap=sum(r['reached_length_cap'] for r in rows))
        repeat=[]
        for row in rows:
            words=re.findall(r'\w+|[^\w\s]',row['text'],flags=re.UNICODE)
            grams=[tuple(words[i:i+3]) for i in range(len(words)-2)]
            repeat.append(0. if not grams else 1-len(set(grams))/len(grams))
        report['groups'][arm]['mean_within_output_repeated_trigram_fraction']=statistics.mean(repeat)
        write(out/f'{arm}-scored.json',[dict(**r,mgpt=s) for r,s in zip(rows,scores)])
        print(arm,report['groups'][arm],flush=True)
    write(out/'generation_metrics.json',report)
    summary_path=out/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        summary['pinned_external_scoring']='completed'
        write(summary_path,summary)


if __name__=='__main__':
    main()
