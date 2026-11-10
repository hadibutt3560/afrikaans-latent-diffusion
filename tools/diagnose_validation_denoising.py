"""Frozen-weight validation diagnostic: real AE latents versus noisy predictions."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers.modeling_outputs import BaseModelOutput
from tools.trace_afrikaans_inference import DiffusionTransformer, GaussianDiffusion, get_latent_model


@torch.no_grad()
def main():
    out=Path(sys.argv[1])
    if out.exists():
        raise FileExistsError(out)
    dp=Path('outputs/cluster-results/diffusion-50k-retry-20260920')
    ap=Path('outputs/cluster-results/afrikaans-56531/best')
    d=SimpleNamespace(**json.loads((dp/'args.json').read_text()))
    a=SimpleNamespace(**json.loads((ap/'args.json').read_text()))
    assert d.objective == 'pred_v' and not d.normalize_latent and not d.class_conditional
    net=DiffusionTransformer(tx_dim=d.tx_dim,tx_depth=d.tx_depth,heads=d.tx_dim//64,
        latent_dim=a.dim_ae,max_seq_len=d.max_seq_len,self_condition=d.self_condition,
        scale_shift=d.scale_shift,dropout=0 if d.disable_dropout else 0.1,
        class_conditional=False,num_classes=0,class_unconditional_prob=d.class_unconditional_prob,
        seq2seq=False,seq2seq_context_dim=768,num_dense_connections=d.num_dense_connections)
    diffusion=GaussianDiffusion(net,max_seq_len=d.max_seq_len,sampling_timesteps=50,
        sampler=d.sampler,train_schedule=d.train_schedule,sampling_schedule=d.sampling_schedule,
        loss_type=d.loss_type,objective=d.objective,train_prob_self_cond=d.train_prob_self_cond,
        seq2seq_unconditional_prob=d.seq2seq_unconditional_prob,scale=d.scale)
    checkpoint=torch.load(str(dp/'model.pt'),map_location='cpu',mmap=True,weights_only=False)
    diffusion.load_state_dict({k[len('ema_model.'):]:v for k,v in checkpoint['ema'].items() if k.startswith('ema_model.')},strict=True)
    del checkpoint
    diffusion.cuda().eval().requires_grad_(False)
    a.enc_dec_model=d.enc_dec_model
    ae,tokenizer,_=get_latent_model(a)
    checkpoint=torch.load(str(ap/'model.pt'),map_location='cpu',mmap=True,weights_only=False)
    ae.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    ae.cuda().eval().requires_grad_(False)
    if '--reverse-recovery' in sys.argv:
        reverse_recovery(out, ae, tokenizer, diffusion)
        return
    if '--interface-check' in sys.argv:
        interface_check(out, ae, tokenizer, diffusion)
        return
    if '--broader' in sys.argv:
        broader(out, ae, tokenizer, diffusion)
        return
    source=Path('data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl')
    examples=[json.loads(line)['text'] for line in source.read_text(encoding='utf8').splitlines()]
    random.Random(31415).shuffle(examples)
    examples=examples[:16]
    out.mkdir()
    results=[]
    torch.manual_seed(31415)
    def measure(latent,batch):
        encoded=BaseModelOutput(last_hidden_state=ae.get_decoder_input(latent))
        labels=batch.input_ids.clone()
        labels[labels == tokenizer.pad_token_id]=-100
        ce=float(ae(encoder_outputs=encoded,labels=labels).loss)
        ids=ae.generate(encoder_outputs=encoded,max_length=128,min_length=5,num_beams=4,
            do_sample=False,no_repeat_ngram_size=3,repetition_penalty=1.2)
        text=tokenizer.decode(ids[0],skip_special_tokens=True)
        return dict(cross_entropy=ce,text=text)
    for index,text in enumerate(examples):
        batch=tokenizer(text,return_tensors='pt',truncation=True,max_length=128).to('cuda')
        reference=tokenizer.decode(batch.input_ids[0],skip_special_tokens=True)
        latent=ae.get_diffusion_latent(ae.get_encoder()(**batch),batch.attention_mask).float()
        noise=torch.randn_like(latent)
        mask=torch.ones(latent.shape[:2],device='cuda',dtype=torch.bool)
        clean=measure(latent,batch)
        results.append(dict(example=index,condition='clean_ae',reference=reference,**clean))
        for value in (0.1,0.3,0.5,0.7,0.9):
            t=torch.tensor([value],device='cuda')
            alpha=diffusion.train_schedule(t).reshape(1,1,1)
            noisy=alpha.sqrt()*latent+(1-alpha).sqrt()*noise
            prediction=diffusion.diffusion_model_predictions(noisy,mask,t).pred_x_start
            target_v=alpha.sqrt()*noise-(1-alpha).sqrt()*latent
            mse=float((prediction-latent).square().mean())
            observed=measure(prediction,batch)
            result=dict(example=index,condition=f'noise_{value}',time=value,alpha=float(alpha),
                reference=reference,latent_mse=mse,noisy_latent_mse=float((noisy-latent).square().mean()),
                v_mse=float((diffusion.diffusion_model_predictions(noisy,mask,t).pred_v-target_v).square().mean()),**observed)
            results.append(result)
        (out/'examples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
        print(f'Completed validation passage {index+1}/16',flush=True)
    report={}
    for condition in sorted({r['condition'] for r in results}):
        rows=[r for r in results if r['condition']==condition]
        report[condition]=dict(mean_passage_cross_entropy=statistics.mean(r['cross_entropy'] for r in rows),
            exact_reconstructions=sum(r['reference']==r['text'] for r in rows))
        if condition != 'clean_ae':
            report[condition].update(mean_latent_mse=statistics.mean(r['latent_mse'] for r in rows),
                mean_noisy_latent_mse=statistics.mean(r['noisy_latent_mse'] for r in rows),
                mean_v_mse=statistics.mean(r['v_mse'] for r in rows))
    summary=dict(status='completed',examples=16,split='validation',selection_seed=31415,
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=report,
        limitations=['One noise draw per passage, 16 passages; diagnostic only.',
        'One denoising prediction without self-conditioning, not a full generation trajectory.',
        'Cross-entropy is mean across passages; reference is truncated tokenized input.',
        'No training or untouched test evaluation.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


@torch.no_grad()
def reverse_recovery(out, ae, tokenizer, diffusion):
    """Known-input recovery is diagnostic, not unconditional generation."""
    out.mkdir()
    path=Path('data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl')
    texts=[json.loads(line)['text'] for line in path.read_text(encoding='utf8').splitlines()]
    random.Random(17320).shuffle(texts)
    diffusion.using_latent_model=True
    diffusion.max_seq_len=32
    diffusion.sampling_timesteps=50
    rows=[]
    for index,text in enumerate(texts[:8]):
        batch=tokenizer(text,return_tensors='pt',truncation=True,max_length=128).to('cuda')
        labels=batch.input_ids.clone()
        labels[labels==tokenizer.pad_token_id]=-100
        latent=ae.get_diffusion_latent(ae.get_encoder()(**batch),batch.attention_mask).float()
        reference=tokenizer.decode(batch.input_ids[0],skip_special_tokens=True)
        def observe(value):
            encoded=BaseModelOutput(last_hidden_state=ae.get_decoder_input(value))
            ce=float(ae(encoder_outputs=encoded,labels=labels).loss)
            ids=ae.generate(encoder_outputs=encoded,max_length=128,min_length=5,num_beams=4,
                do_sample=False,no_repeat_ngram_size=3,repetition_penalty=1.2)
            return dict(ce=ce,text=tokenizer.decode(ids[0],skip_special_tokens=True),
                latent_mse=float((value-latent).square().mean()))
        rows.append(dict(example=index,reference=reference,condition='clean',**observe(latent)))
        torch.manual_seed(17320+index)
        noise=torch.randn_like(latent)
        for start in (.3,.7,.9):
            t=torch.tensor([start],device='cuda')
            alpha=diffusion.train_schedule(t).reshape(1,1,1)
            noisy=alpha.sqrt()*latent+(1-alpha).sqrt()*noise
            first=diffusion.diffusion_model_predictions(noisy,torch.ones(1,32,device='cuda',dtype=torch.bool),t).pred_x_start
            rows.append(dict(example=index,reference=reference,condition='single_prediction',start=start,**observe(first)))
            for sampler in ('ddpm','ddim'):
                def pairs(batch,device,**kwargs):
                    times=torch.linspace(start,0,51,device=device).expand(batch,-1)
                    return [(times[:,i],times[:,i+1]) for i in range(50)]
                diffusion.get_sampling_timesteps=pairs
                torch.manual_seed(27320+index)
                recovered,_=getattr(diffusion,f'{sampler}_sample')(latent.shape,None,None,None,None,z_t=noisy.clone())
                rows.append(dict(example=index,reference=reference,condition=sampler,start=start,**observe(recovered)))
            (out/'examples.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
        print(f'Reverse recovery passage {index+1}/8',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',selection_seed=17320,
        passages=8,steps=50,split='validation',data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        limits=['Known-input recovery from forward-corrupted latents, not unconditional generation.',
            'Same initial Gaussian noise within each passage; samplers differ in reverse stochasticity.',
            'Original self-conditioning and decoding constraints retained. Small diagnostic sample.'],
        results=rows),ensure_ascii=False,indent=2),encoding='utf8')


@torch.no_grad()
def interface_check(out, ae, tokenizer, diffusion):
    out.mkdir()
    path=Path('data/nchlt_afrikaans/processed/stage_5_7/valid.jsonl')
    texts=[json.loads(line)['text'] for line in path.read_text(encoding='utf8').splitlines()]
    random.Random(16180).shuffle(texts)
    rows=[]
    for index,text in enumerate(texts[:8]):
        short=tokenizer(text,return_tensors='pt',truncation=True,max_length=128).to('cuda')
        padded=tokenizer(text,return_tensors='pt',truncation=True,padding='max_length',max_length=128).to('cuda')
        latent=ae.get_diffusion_latent(ae.get_encoder()(**short),short.attention_mask).float()
        other=ae.get_diffusion_latent(ae.get_encoder()(**padded),padded.attention_mask).float()
        labels=short.input_ids.clone()
        labels[labels==tokenizer.pad_token_id]=-100
        def measure(value):
            return float(ae(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(value)),labels=labels).loss)
        item=dict(example=index,tokens=int(short.attention_mask.sum()),
            padding_latent_max_abs_difference=float((latent-other).abs().max()),
            mean_latent_norm=float(latent.norm(dim=-1).mean()),clean_ce=measure(latent),perturbations=[])
        torch.manual_seed(16180+index)
        noise=torch.randn_like(latent)
        for std in (.01,.05,.1,.2):
            # Preserve the AE norm to distinguish directional sensitivity from norm drift.
            altered=torch.nn.functional.normalize(latent+std*noise,dim=-1)*8
            item['perturbations'].append(dict(std=std,ce=measure(altered),mse=float((altered-latent).square().mean())))
        rows.append(item)
        print(f'Interface check {index+1}/8',flush=True)
    report=dict(status='completed',selection_seed=16180,split='validation',rows=rows,
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        decoder_start_token_id=ae.config.decoder_start_token_id,
        eos_token_id=ae.config.eos_token_id,pad_token_id=ae.config.pad_token_id,
        tokenizer_pad_token_id=tokenizer.pad_token_id,tokenizer_eos_token_id=tokenizer.eos_token_id,
        max_padding_latent_difference=max(r['padding_latent_max_abs_difference'] for r in rows),
        mean_clean_ce=statistics.mean(r['clean_ce'] for r in rows),
        mean_perturbed_ce={str(std):statistics.mean(r['perturbations'][i]['ce'] for r in rows)
                           for i,std in enumerate((.01,.05,.1,.2))},
        limits=['Eight validation passages; mean passage CE, not population/token-weighted CE.',
            'Norm-preserving Gaussian perturbations are not actual diffusion prediction errors.',
            'Padding invariance tested in FP32 evaluation mode; no training or generation.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


@torch.no_grad()
def broader(out, ae, tokenizer, diffusion):
    out.mkdir()
    results=[]
    hashes={}
    for split in ('train','valid'):
        source=Path(f'data/nchlt_afrikaans/processed/stage_5_7/{split}.jsonl')
        hashes[split]=hashlib.sha256(source.read_bytes()).hexdigest()
        texts=[json.loads(line)['text'] for line in source.read_text(encoding='utf8').splitlines()]
        random.Random(27182).shuffle(texts)
        selected={k:[] for k in ('short','medium','long')}
        for text in texts:
            size=len(tokenizer(text).input_ids)
            bucket='short' if size<=48 else 'medium' if size<=96 else 'long'
            if len(selected[bucket])<8:
                selected[bucket].append((text,size))
            if all(len(v)==8 for v in selected.values()):
                break
        for bucket, passages in selected.items():
            for index,(text,size) in enumerate(passages):
                batch=tokenizer(text,return_tensors='pt',truncation=True,max_length=128).to('cuda')
                labels=batch.input_ids.clone()
                labels[labels==tokenizer.pad_token_id]=-100
                tokens=int((labels!=-100).sum())
                latent=ae.get_diffusion_latent(ae.get_encoder()(**batch),batch.attention_mask).float()
                mask=torch.ones(latent.shape[:2],device='cuda',dtype=torch.bool)
                def ce(value):
                    return float(ae(encoder_outputs=BaseModelOutput(last_hidden_state=ae.get_decoder_input(value)),labels=labels).loss)
                base=dict(split=split,bucket=bucket,example=index,original_tokens=size,tokens=tokens,truncated=size>128)
                results.append(dict(**base,condition='clean',cross_entropy=ce(latent)))
                for draw in range(3):
                    torch.manual_seed(27182+index*3+draw)
                    noise=torch.randn_like(latent)
                    for value in (.1,.3,.5,.7,.9):
                        t=torch.tensor([value],device='cuda')
                        alpha=diffusion.train_schedule(t).reshape(1,1,1)
                        noisy=alpha.sqrt()*latent+(1-alpha).sqrt()*noise
                        first=diffusion.diffusion_model_predictions(noisy,mask,t)
                        target=alpha.sqrt()*noise-(1-alpha).sqrt()*latent
                        results.append(dict(**base,condition=f'noise_{value}',draw=draw,
                            cross_entropy=ce(first.pred_x_start),
                            latent_mse=float((first.pred_x_start-latent).square().mean()),
                            noisy_mse=float((noisy-latent).square().mean()),
                            v_mse=float((first.pred_v-target).square().mean())))
                (out/'measurements.json').write_text(json.dumps(results,indent=2),encoding='utf8')
                print(f'Completed {split} {bucket} passage {index+1}/8',flush=True)
    groups=[]
    for split in ('train','valid'):
        for bucket in ('all','short','medium','long'):
            for condition in sorted({r['condition'] for r in results}):
                rows=[r for r in results if r['split']==split and r['condition']==condition and (bucket=='all' or r['bucket']==bucket)]
                item=dict(split=split,bucket=bucket,condition=condition,observations=len(rows),
                    token_weighted_cross_entropy=sum(r['cross_entropy']*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows))
                if condition!='clean':
                    item.update({k:statistics.mean(r[k] for r in rows) for k in ('latent_mse','noisy_mse','v_mse')})
                groups.append(item)
    summary=dict(status='completed',passages_per_split=24,noise_draws=3,selection_seed=27182,
        data_sha256=hashes,results=groups,limitations=[
        'Length-balanced diagnostic sample, not a population estimate; 8 passages per length group.',
        'Frozen EMA weights, one prediction without self-conditioning; not full diffusion generation.',
        'Teacher-forced cross-entropy measures reconstruction, not generated-text quality.',
        'Three noise draws reuse each passage; these are not independent passages.',
        'Long inputs are truncated at 128 tokens. No test data or training used.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
