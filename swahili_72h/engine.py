"""Single GPU primitives. Imports only frozen project models and offline HF assets."""
import contextlib
import copy
import gc
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch
from transformers import MT5Config, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from common import read, write, append, sha, identity
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from latent_models.t5_latent_model import MT5ForConditionalGenerationLatent
from model.diffusion_transformer import DiffusionTransformer
from gaussian_diffusion import GaussianDiffusion

STOP=False
DEADLINE=float('inf')


def stopping(*_):
    global STOP
    STOP=True


for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stopping)
if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,stopping)


def check(margin=0):
    if STOP or time.time()+margin>=DEADLINE:raise InterruptedError('Signal or internal stage deadline')


def seed(n):
    random.seed(n);np.random.seed(n%(2**32));torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    if r['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


@contextlib.contextmanager
def isolated(n):
    r=rng();seed(n)
    try:yield
    finally:restore(r)


def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def clean():
    gc.collect();torch.cuda.empty_cache()


def amp(c):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['precision']=='bf16')


def setup(c):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; never run on login node')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if c['precision']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported')
    x=torch.randn(64,64,device='cuda');assert torch.isfinite(x@x.T).all()
    return dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),memory=torch.cuda.get_device_properties(0).total_memory)


def tokenizer(root):return T5Tokenizer.from_pretrained(str(root/'assets/mt5-base'),local_files_only=True,legacy=True)


def ae_model(root,c,checkpoint=None):
    seed(c['seed']); config=MT5Config.from_pretrained(str(root/'assets/mt5-base'),local_files_only=True)
    # Transformers 5.5 overrides the serialized mT5 tying flag. Preserve the actual
    # pretrained architecture explicitly; cluster uses pinned Transformers 4.38.2.
    config.tie_word_embeddings=read(root/'assets/mt5-base/config.json')['tie_word_embeddings']
    ae=MT5ForConditionalGenerationLatent(config,num_encoder_latents=c['latent_length'],num_decoder_latents=c['latent_length'],
        dim_ae=c['latent_dim'],num_layers=2,l2_normalize_latents=True)
    ae.config.tie_word_embeddings=False
    ae.encoder.set_input_embeddings(ae.shared);ae.decoder.set_input_embeddings(ae.shared)
    if ae.lm_head.weight is ae.shared.weight:
        ae.lm_head=torch.nn.Linear(config.d_model,config.vocab_size,bias=False)
    assert ae.lm_head.weight is not ae.shared.weight
    if checkpoint:
        state=load(checkpoint);ae.load_state_dict(state['model'],strict=True);del state
    else:
        base=load(root/'assets/mt5-base/pytorch_model.bin');missing,unexpected=ae.load_state_dict(base,strict=False)
        if unexpected or any(not k.startswith('perceiver_ae.') for k in missing):raise ValueError((missing,unexpected))
        if not torch.equal(ae.shared.weight,base['shared.weight']) or not torch.equal(ae.lm_head.weight,base['lm_head.weight']):
            raise ValueError('Pretrained mT5 embeddings/output head changed during loading')
        del base
    ae.requires_grad_(False)
    ae.perceiver_ae.requires_grad_(True)
    # Swahili reconstruction trains the complete decoder, but not tied embeddings or LM head.
    ae.decoder.requires_grad_(True);ae.shared.requires_grad_(False);ae.lm_head.requires_grad_(False)
    return ae.cuda()


def diffusion(c,name):
    s=c['models'][name]
    net=DiffusionTransformer(tx_dim=s['width'],tx_depth=s['depth'],heads=s['width']//64,latent_dim=c['latent_dim'],
        max_seq_len=c['max_length'],self_condition=True,scale_shift=True,dropout=.1,num_dense_connections=s['dense_connections'])
    m=GaussianDiffusion(net,max_seq_len=c['latent_length'],objective='pred_v',loss_type='l2',train_schedule='cosine',
                        sampler=c['sampler'],sampling_timesteps=c['sampling_steps'],train_prob_self_cond=.5)
    m.using_latent_model=True;m.l2_normalize=True;m.normalize_sampling_self_condition=c['normalize_sampling_self_condition']
    return m.cuda()


def token_batch(values,ids):
    x=torch.as_tensor(np.array(values[ids],copy=True),device='cuda',dtype=torch.long)
    mask=x.ne(0);labels=x.masked_fill(~mask,-100)
    return x,mask,labels


def ae_forward(model,values,ids):
    x,mask,labels=token_batch(values,ids)
    with torch.no_grad():enc=model.get_encoder()(input_ids=x,attention_mask=mask)
    z=model.get_diffusion_latent(enc,mask)
    out=model(labels=labels,encoder_outputs=BaseModelOutput(last_hidden_state=model.get_decoder_input(z)))
    return out,labels


def loss(model,values,ids,c,kind):
    with amp(c):
        if kind=='ae':return ae_forward(model,values,ids)[0].loss
        z=torch.as_tensor(np.array(values[ids],copy=True),device='cuda')
        return model(z,torch.ones(z.shape[:2],dtype=torch.bool,device='cuda'),class_id=None)


@torch.no_grad()
def validate(model,values,c,kind,count=None):
    model.eval();n=min(len(values),count or c['validation_examples']);batch=c[kind]['micro_batch']
    ids=np.random.default_rng(c['validation_seed']).permutation(len(values))[:n]
    repeats=1 if kind=='ae' else c['validation_repeats'];total=weight=correct=tokens=0
    with isolated(c['validation_seed']):
        for repeat in range(repeats):
            seed(c['validation_seed']+repeat)
            for start in range(0,n,batch):
                check();ix=ids[start:start+batch]
                if kind=='ae':
                    with amp(c):out,labels=ae_forward(model,values,ix)
                    value=out.loss;num=int(labels.ne(-100).sum());correct+=int(((out.logits.argmax(-1)==labels)&labels.ne(-100)).sum());tokens+=num
                else:value=loss(model,values,ix,c,kind);num=len(ix)
                if not torch.isfinite(value):raise FloatingPointError('Validation loss')
                total+=float(value)*num;weight+=num
    return dict(loss=total/weight,token_accuracy=correct/tokens if tokens else None,examples=n,repeats=repeats)


@torch.no_grad()
def reconstruct(model,values,c,tok,count=32):
    from metrics import chrf
    model.eval();items=[]
    ids=np.random.default_rng(c['validation_seed']).permutation(len(values))[:count]
    for i in ids:
        check();x,mask,_=token_batch(values,[i])
        z=model.get_diffusion_latent(model.get_encoder()(input_ids=x,attention_mask=mask),mask)
        out=model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=model.get_decoder_input(z)),
                           max_length=c['max_length'],do_sample=False,num_beams=1)
        ref=tok.decode(x[0],skip_special_tokens=True);pred=tok.decode(out[0],skip_special_tokens=True)
        items.append(dict(row=int(i),reference=ref,reconstruction=pred,chrf=chrf(ref,pred)))
    return dict(mean_chrf=sum(x['chrf'] for x in items)/len(items),rows=items)


def lr_at(step,spec):
    # Inverse-square-root schedule has no wall-clock-dependent horizon.
    warm=spec['warmup'];return spec['lr']*min((step+1)/warm,max(.05,math.sqrt(warm/max(step+1,warm))))


def train(model,train_values,valid_values,c,kind,folder,run_seed,contract,steps=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);s=c[kind];seed(run_seed)
    contract=dict(inputs=contract,kind=kind,seed=run_seed,configuration=c,
                  parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()})
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW(params.values(),lr=s['lr'],weight_decay=c['weight_decay'])
    ema={n:p.detach().clone() for n,p in params.items()} if s['ema'] else {}
    state=dict(step=0,best=None,patience_best=None,stale=0,validated=-1,seconds=0,optimization_seconds=0,validation_seconds=0,status='running')
    latest=folder/'latest.pt'
    if latest.exists():
        ck=load(latest)
        if ck['contract']!=contract or ck['scheduler']['spec']!=s:raise ValueError('Resume contract changed')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);state=ck['state']
        ema={n:v.cuda() for n,v in ck['ema'].items()};restore(ck['rng']);del ck
    if state['status'] in ('converged','max_steps'):return state
    began=time.monotonic();prior=state['seconds'];last_save=began;initial=state['step'];order=None;epoch_old=None;timings=[]

    def checkpoint(path):
        nonlocal last_save
        state['seconds']=prior+time.monotonic()-began
        save(path,dict(schema=1,contract=contract,model=model.state_dict(),optimizer=opt.state_dict(),ema=ema,rng=rng(),
                       scheduler=dict(kind='warmup_inverse_sqrt_floor',spec=s,next_step=state['step'],next_lr=lr_at(state['step'],s)),state=dict(state)))
        last_save=time.monotonic();write(folder/'status.json',state)

    def validation():
        validation_start=time.monotonic()
        backup={}
        if ema:
            with torch.no_grad():
                for n,p in params.items():backup[n]=p.detach().clone();p.copy_(ema[n])
        try:metrics=validate(model,valid_values,c,kind)
        finally:
            if backup:
                with torch.no_grad():
                    for n,p in params.items():p.copy_(backup[n])
        value=metrics['loss'];state['validated']=state['step']
        state['validation_seconds']=state.get('validation_seconds',0)+time.monotonic()-validation_start
        improved=state['best'] is None or value<state['best']
        if improved:state['best']=value
        if state['patience_best'] is None or value<state['patience_best']*(1-s['relative_delta']):state['patience_best']=value;state['stale']=0
        else:state['stale']+=1
        append(folder/'metrics.jsonl',dict(kind='validation',step=state['step'],**metrics))
        # Best and latest both retain complete resume state; inference explicitly uses EMA.
        if improved:checkpoint(folder/'best.pt')
        checkpoint(latest)

    if not latest.exists():checkpoint(latest)
    try:
        if state['validated']<0:validation()
        elif state['step']%s['validate_every']==0 and state['validated']!=state['step']:validation()
        while state['step']<s['max_steps']:
            if STOP or time.time()+180>=DEADLINE:break
            if steps is not None and state['step']-initial>=steps:break
            tick=time.monotonic();batch=s['micro_batch']*s['accumulation'];spe=math.ceil(len(train_values)/batch)
            epoch,pos=divmod(state['step'],spe)
            if epoch!=epoch_old:order=np.random.default_rng(run_seed+epoch*1000003).permutation(len(train_values));epoch_old=epoch
            ids=order[pos*batch:(pos+1)*batch];model.train()
            if kind=='ae':model.get_encoder().eval()
            opt.zero_grad(set_to_none=True);learning_rate=lr_at(state['step'],s)
            for group in opt.param_groups:group['lr']=learning_rate
            value=0.
            for start in range(0,len(ids),s['micro_batch']):
                micro=ids[start:start+s['micro_batch']]
                v=loss(model,train_values,micro,c,kind)*len(micro)/len(ids)
                if not torch.isfinite(v):raise FloatingPointError('Training loss')
                v.backward();value+=float(v.detach())
            grad=torch.nn.utils.clip_grad_norm_(list(params.values()),c['clip_grad_norm'],error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for n,p in params.items():
                    if not torch.isfinite(p).all():raise FloatingPointError('Optimizer produced non-finite parameters; last checkpoint retained')
                    if ema:ema[n].lerp_(p,1-s['ema'])
            state['step']+=1;torch.cuda.synchronize();dt=time.monotonic()-tick;timings.append(dt)
            state['optimization_seconds']=state.get('optimization_seconds',0)+dt
            if state['step']%10==0 or steps:
                row=dict(kind='train',step=state['step'],loss=value,gradient=float(grad),lr=learning_rate,seconds=dt,
                         examples_per_second=len(ids)/dt,peak_vram=torch.cuda.max_memory_allocated())
                append(folder/'metrics.jsonl',row);print(row,flush=True)
            if state['step']%s['validate_every']==0:
                checkpoint(latest)
                if time.time()+300>=DEADLINE:break
                validation()
                if state['step']>=s['minimum_steps'] and state['stale']>=s['patience']:state['status']='converged';break
            if state['step']%c['checkpoint_steps']==0 or time.monotonic()-last_save>=c['checkpoint_seconds']:checkpoint(latest)
        if not STOP and time.time()+300<DEADLINE and state['validated']!=state['step']:validation()
        if state['step']>=s['max_steps']:state['status']='max_steps'
        elif state['status']!='converged':state['status']='paused'
        checkpoint(latest)
    except InterruptedError:
        # Validation is RNG-isolated and EMA restored before saving.
        state['status']='paused';checkpoint(latest)
        if STOP:raise
    state['timings']=timings
    return state


def inference_checkpoint(model,path):
    ck=load(path);model.load_state_dict(ck['model'])
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in ck['ema']:p.copy_(ck['ema'][n])
    del ck;return model.eval().requires_grad_(False)


@torch.no_grad()
def generate(model,ae,tok,c,number,seed_start):
    output=[]
    with isolated(seed_start):
        for i in range(number):
            check();seed(seed_start+i)
            z,_=model.sample(1,torch.tensor([c['latent_length']],device='cuda'))
            if not torch.isfinite(z).all():raise FloatingPointError('Generated latent')
            ids=ae.generate(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(z)),
                max_length=c['max_length'],min_length=5,num_beams=4,do_sample=False,no_repeat_ngram_size=3,repetition_penalty=1.2)[0]
            raw=tok.decode(ids,skip_special_tokens=False)
            output.append(dict(seed=seed_start+i,text=tok.decode(ids,skip_special_tokens=True).strip(),
                               tokens=len(ids),eos=bool((ids[1:]==tok.eos_token_id).any()),sentinel='<extra_id_' in raw))
    return output


def scorer(root):
    tok=GPT2Tokenizer.from_pretrained(str(root/'assets/mgpt'),local_files_only=True)
    model=GPT2LMHeadModel.from_pretrained(str(root/'assets/mgpt'),local_files_only=True,torch_dtype=torch.float32)
    return model.cuda().eval().requires_grad_(False),tok


@torch.no_grad()
def score(model,tok,text,context=512):
    tokens=tok.encode(text,add_special_tokens=False)
    if not tokens:return dict(empty=True,nll=0.,tokens=0,perplexity=None)
    bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if bos is None:raise ValueError('Scorer lacks BOS/EOS token')
    ids=torch.tensor([[bos]+tokens],device='cuda');previous=1;total=0.;count=0
    for begin in range(0,ids.size(1),context//2):
        check();end=min(begin+context,ids.size(1));x=ids[:,begin:end];labels=x.clone();labels[:,:max(1,previous-begin)]=-100
        n=int(labels[:,1:].ne(-100).sum())
        if n:
            value=float(model(input_ids=x,labels=labels).loss)
            if not math.isfinite(value):raise FloatingPointError('Scorer')
            total+=value*n;count+=n
        previous=end
        if end==ids.size(1):break
    return dict(empty=False,nll=total,tokens=count,perplexity=math.exp(min(700,total/count)))
