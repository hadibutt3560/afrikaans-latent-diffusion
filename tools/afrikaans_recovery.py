"""Bounded, reproducible autoencoder diagnostics; never touches the test split for scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/nchlt-afrikaans/recovery'
BASE = ROOT / 'outputs/nchlt-afrikaans/autoencoder-6k'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preserve():
    backup = OUT / 'baseline-step6000'
    backup.mkdir(parents=True, exist_ok=True)
    for name in ('model.pt', 'args.json', 'autoencoder-valid-step-6000.json'):
        source, target = BASE / name, backup / name
        if not target.exists():
            shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise RuntimeError(f'Backup differs: {name}')
    (backup / 'sha256.json').write_text(json.dumps({
        name: digest(backup / name) for name in ('model.pt', 'args.json')
    }, indent=2))
    return backup


def command(output, mode, steps, init, tiny=False, width=64):
    cmd = [sys.executable, '-u', 'train_latent_model.py',
           '--dataset_name', 'nchlt_afrikaans', '--enc_dec_model', 'google/mt5-base',
           '--max_seq_len', '128', '--train_batch_size', '1', '--eval_batch_size', '1',
           '--gradient_accumulation_steps', '8', '--num_encoder_latents', '32',
           '--num_decoder_latents', '32', '--dim_ae', str(width), '--num_layers', '3',
           '--l2_normalize_latents', '--learning_rate', '0.0001',
           '--lr_schedule', 'constant_with_warmup', '--lr_warmup_steps', '25',
           '--num_train_steps', str(steps), '--eval_every', '250',
           '--validation_max_examples', '32' if tiny else '128',
           '--skip_expensive_metrics', '--fail_on_validation_error',
           '--dataloader_num_workers', '0', '--mixed_precision', 'bf16',
           '--lm_mode', mode, '--output_dir', str(output), '--seed', '42']
    if init:
        cmd += ['--init_dir', str(init)]
    if tiny:
        cmd += ['--train_subset_size', '32', '--overfit_subset']
    else:
        cmd += ['--validation_shuffle']
    return cmd


def run(name, cmd):
    directory = OUT / name
    if directory.exists():
        raise RuntimeError(f'Refusing to overwrite {directory}')
    directory.mkdir(parents=True)
    (directory / 'command.json').write_text(json.dumps(cmd, indent=2))
    print(f'START {name}', flush=True)
    with (directory / 'console.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'FINISHED {name}', flush=True)


def summarize():
    result = {}
    for directory in OUT.iterdir():
        files = list(directory.glob('autoencoder-valid-step-*.json'))
        if files:
            latest = max(files, key=lambda p: int(p.stem.rsplit('-', 1)[1]))
            result[directory.name] = json.loads(latest.read_text())
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['diagnostic', 'pilot', 'long', 'capacity', 'summary'], default='diagnostic')
    parser.add_argument('--mode', choices=['freeze', 'decoder_last'], default='decoder_last')
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--init-dir', type=Path, help='Explicit weights-only initialization for a longer run.')
    args = parser.parse_args()
    os.environ.update(WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
                      HF_HOME='/home/scientist/.cache/huggingface', PYTHONUNBUFFERED='1')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'summary':
        summarize()
        return
    backup = preserve()
    if args.phase == 'capacity':
        for width in (64, 128):
            name = f'capacity-{args.mode}-width{width}-{args.steps}'
            run(name, command(OUT / name, args.mode, args.steps, None, width=width))
            summarize()
        return
    modes = ['freeze', 'decoder_last'] if args.phase == 'diagnostic' else [args.mode]
    for mode in modes:
        name = f'{args.phase}-{mode}-{args.steps}'
        run(name, command(OUT / name, mode, args.steps, args.init_dir or backup,
                          tiny=args.phase == 'diagnostic'))
        summarize()


if __name__ == '__main__':
    main()
