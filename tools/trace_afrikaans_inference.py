"""Trace clean-latent predictions using existing weights; never train or overwrite runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers.modeling_outputs import BaseModelOutput
from diffusion.text_denoising_diffusion import GaussianDiffusion
from diffusion.constant import generate_kwargs
from model.diffusion_transformer import DiffusionTransformer
from latent_models.latent_utils import get_latent_model
from tools.inference_trace_utils import capture_predictions, annotate


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diffusion-dir', type=Path, required=True)
    parser.add_argument('--autoencoder-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--every', type=int, default=25)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--decode-batch-size', type=int, default=1)
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'])
    parser.add_argument('--normalize-sampling-self-condition', action='store_true')
    parser.add_argument('--decode-max-length', type=int, default=64)
    parser.add_argument('--project-decoder-latents', action='store_true',
                        help='Project only decoder inputs to the AE training norm; leave sampling unchanged')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory; existing results are protected')
    if args.every < 1 or args.decode_batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require positive observation interval and distinct seeds')
    d = SimpleNamespace(**json.loads((args.diffusion_dir / 'args.json').read_text()))
    a = SimpleNamespace(**json.loads((args.autoencoder_dir / 'args.json').read_text()))
    if args.decode_max_length < 5:
        parser.error('Decoder length must be at least 5')
    if args.project_decoder_latents and not a.l2_normalize_latents:
        parser.error('Projection requires an AE trained with normalized latents')
    provenance = json.loads((args.diffusion_dir / 'latent_model_provenance.json').read_text())
    if not provenance.get('using_latent_model') or not d.using_latent_model:
        raise ValueError('Checkpoint does not confirm compressed latent diffusion')
    if (a.dim_ae, a.num_encoder_latents) != (provenance['latent_dimension'], provenance['latent_length']):
        raise ValueError('Autoencoder dimensions do not match provenance')
    sampler = args.sampler or d.sampler
    if sampler not in ('ddpm', 'ddim') or d.class_conditional or d.normalize_latent:
        raise ValueError('This diagnostic supports unconditional, unnormalized DDPM/DDIM only')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    steps = args.steps if args.steps is not None else d.sampling_timesteps
    if steps < 1:
        parser.error('Steps must be positive')
    checkpoint = torch.load(str(args.diffusion_dir / 'model.pt'), map_location='cpu', mmap=True, weights_only=False)
    if not checkpoint.get('using_latent_model'):
        raise ValueError('Checkpoint metadata disagrees with config')
    net = DiffusionTransformer(tx_dim=d.tx_dim, tx_depth=d.tx_depth, heads=d.tx_dim // 64,
        latent_dim=a.dim_ae, max_seq_len=d.max_seq_len, self_condition=d.self_condition,
        scale_shift=d.scale_shift, dropout=0 if d.disable_dropout else 0.1,
        class_conditional=False, num_classes=0, class_unconditional_prob=d.class_unconditional_prob,
        seq2seq=False, seq2seq_context_dim=768, num_dense_connections=d.num_dense_connections)
    diffusion = GaussianDiffusion(net, max_seq_len=d.max_seq_len, sampling_timesteps=steps,
        sampler=sampler, train_schedule=d.train_schedule, sampling_schedule=d.sampling_schedule,
        loss_type=d.loss_type, objective=d.objective, train_prob_self_cond=d.train_prob_self_cond,
        seq2seq_unconditional_prob=d.seq2seq_unconditional_prob, scale=d.scale)
    ema = {key[len('ema_model.'):]: val for key, val in checkpoint['ema'].items() if key.startswith('ema_model.')}
    if not ema:
        raise ValueError('No EMA weights; refusing to substitute training weights silently')
    diffusion.load_state_dict(ema, strict=True)
    training_step = checkpoint['step']
    diffusion.max_seq_len = a.num_encoder_latents
    diffusion.using_latent_model = True
    diffusion.l2_normalize = a.l2_normalize_latents
    if args.normalize_sampling_self_condition and not (d.self_condition and a.l2_normalize_latents):
        parser.error('Self-conditioning alignment requires normalized AE latents and self-conditioning')
    diffusion.normalize_sampling_self_condition = args.normalize_sampling_self_condition
    diffusion.cuda().eval().requires_grad_(False)
    del checkpoint, ema
    a.enc_dec_model = d.enc_dec_model
    autoencoder, tokenizer, _ = get_latent_model(a)
    ae_checkpoint = torch.load(str(args.autoencoder_dir / 'model.pt'), map_location='cpu', mmap=True, weights_only=False)
    autoencoder.load_state_dict(ae_checkpoint['model'], strict=True)
    ae_step = ae_checkpoint['step']
    del ae_checkpoint
    autoencoder.cuda().eval().requires_grad_(False)
    # Match current repository beam decoding. Record it rather than claiming remote-code identity.
    decoding = dict(generate_kwargs['beam'])
    decoding['max_length'] = args.decode_max_length
    args.output_dir.mkdir(parents=True)
    metadata = dict(status='running', diffusion_training_step=training_step, autoencoder_step=ae_step,
        sampler=sampler, inference_steps=steps, observe_every=args.every, seeds=args.seeds,
        project_decoder_latents=args.project_decoder_latents,
        normalize_sampling_self_condition=args.normalize_sampling_self_condition,
        weights='EMA', batch_size=1, decode_batch_size=args.decode_batch_size,
        precision='float32', gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, decoding=decoding,
        diffusion_sha256=sha256(args.diffusion_dir / 'model.pt'),
        autoencoder_sha256=sha256(args.autoencoder_dir / 'model.pt'),
        autoencoder_pairing='Provenance source path and architecture consistent with downloaded job 56531 best. Remote content hash unavailable.',
        source_provenance=provenance,
        limitations=['Decoded predicted clean latents, not raw noisy latents.',
                    'Different batch size/precision/environment can change outputs from cluster sample files.',
                    'Intermediate predictions are diagnostics, not independent training checkpoints.',
                    'No external quality score or human fluency rating; stability does not establish quality.',
                    'Changes between unobserved steps cannot be counted when every > 1.'])
    write_json(args.output_dir / 'metadata.json', metadata)
    all_rows, seed_summaries = [], []
    start = time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with capture_predictions(diffusion, args.every) as snapshots:
            final, _ = diffusion.sample(batch_size=1, length=torch.tensor([a.num_encoder_latents], device='cuda'))
        if not torch.equal(final.float().cpu(), snapshots[-1][1]):
            raise AssertionError('Last clean prediction differs from final sampled latent')
        rows = []
        for offset in range(0, len(snapshots), args.decode_batch_size):
            group = snapshots[offset:offset + args.decode_batch_size]
            latent = torch.cat([item[1] for item in group])
            if not torch.isfinite(latent).all():
                raise RuntimeError(f'Non-finite latents in decoding group {offset}')
            raw_norm = latent.norm(dim=-1).mean(dim=-1)
            if args.project_decoder_latents:
                latent = torch.nn.functional.normalize(latent, dim=-1) * (latent.shape[-1] ** 0.5)
            encoded = BaseModelOutput(last_hidden_state=autoencoder.get_decoder_input(latent.cuda()))
            ids = autoencoder.generate(encoder_outputs=encoded, attention_mask=None, **decoding)
            for item_index, ((step, _), tokens) in enumerate(zip(group, ids)):
                eos = (tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                token_count = int(eos[0]) + 1 if len(eos) else len(tokens)
                text = tokenizer.decode(tokens[:token_count], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                rows.append(dict(seed=seed, step=step, text=text, decoder_tokens=token_count,
                                 raw_latent_mean_norm=float(raw_norm[item_index]),
                                 ae_target_norm=a.dim_ae ** 0.5,
                                 reached_length_cap=bool(token_count >= decoding['max_length']),
                                 ended_with_eos=bool(len(eos))))
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        rows = annotate(rows)
        with (args.output_dir / 'trajectories.jsonl').open('a', encoding='utf8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        all_rows.extend(rows)
        seed_summaries.append(dict(seed=seed, observed_points=len(rows),
            changed_transitions=sum(bool(x['changed_from_previous']) for x in rows),
            compared_transitions=len(rows)-1, revisits=sum(x['revisits_earlier_text'] for x in rows),
            final_text=rows[-1]['text']))
    by_step = []
    for step in sorted({r['step'] for r in all_rows}):
        subset = [r for r in all_rows if r['step'] == step]
        comparisons = [r for r in subset if r['changed_from_previous'] is not None]
        by_step.append(dict(step=step, samples=len(subset), compared=len(comparisons),
            changed=sum(bool(r['changed_from_previous']) for r in comparisons),
            fraction_changed=None if not comparisons else sum(r['changed_from_previous'] for r in comparisons)/len(comparisons),
            mean_word_edits=None if not comparisons else sum(r['word_edits_from_previous'] for r in comparisons)/len(comparisons)))
    with (args.output_dir / 'changes_by_step.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(by_step[0]))
        writer.writeheader(); writer.writerows(by_step)
    metadata.update(status='completed', elapsed_sampling_and_decoding_seconds=time.monotonic()-start,
                    seed_summaries=seed_summaries, quality_improvement='Not measured; inspect texts and add a validated quality scorer.')
    write_json(args.output_dir / 'metadata.json', metadata)
    lines = ['# Inference trajectory diagnostic', '',
             f'Training checkpoint: {training_step}; {sampler.upper()} inference steps: {steps}.',
             f'Observed every {args.every} steps plus first/final; {len(args.seeds)} trajectories. No retraining.',
             'Word edits include punctuation, use case-sensitive Unicode tokenization, and are not fluency scores.', '']
    for seed in args.seeds:
        lines.extend([f'## Seed {seed}', ''])
        for row in all_rows:
            if row['seed'] == seed:
                lines.extend([f"**Step {row['step']}** (word edits since previous observation: {row['word_edits_from_previous']})", row['text'], ''])
    (args.output_dir / 'READ_RESULTS.md').write_text('\n'.join(lines), encoding='utf8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
