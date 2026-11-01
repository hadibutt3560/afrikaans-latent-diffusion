"""One-allocation offline GPU check and Afrikaans continuation."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_environment():
    from importlib.metadata import version
    expected = {'torch': '2.7.1', 'transformers': '4.38.2', 'accelerate': '0.27.2',
                'datasets': '2.17.1', 'numpy': '1.26.4'}
    actual = {name: version(name) for name in expected}
    print(json.dumps(actual, indent=2), flush=True)
    for name, required in expected.items():
        if actual[name].split('+')[0] != required:
            raise RuntimeError(f'{name}: expected {required}, found {actual[name]}')
    import torch
    if torch.version.cuda != '12.8':
        raise RuntimeError(f'Expected CUDA 12.8 PyTorch build, found {torch.version.cuda}')
    from latent_models.latent_finetuning import Trainer
    for path in ['inputs/afrikaans-pilot/model.pt', 'inputs/afrikaans-pilot/args.json',
                 'models/mt5-base/config.json', 'models/mt5-base/pytorch_model.bin',
                 'models/mt5-base/spiece.model']:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    for split in ('train', 'valid', 'test'):
        if not (ROOT / f'data/nchlt_afrikaans/processed/stage_5_7/{split}.jsonl').is_file():
            raise FileNotFoundError(split)
    checkpoint = torch.load(ROOT / 'inputs/afrikaans-pilot/model.pt', map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('step') != 500 or not {'model', 'opt', 'scheduler'}.issubset(checkpoint):
        raise RuntimeError('Unexpected pilot checkpoint; inspect before submitting')
    print('CPU environment, required files and checkpoint compatibility checked.', flush=True)


def main():
    os.chdir(ROOT)
    os.environ.update(WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    check_environment()
    if '--check-only' in sys.argv:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA GPU; training will not run on CPU.')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    torch.ones(16, device='cuda').square().sum().item()
    from train_latent_model import build_parser, merge_saved_args, main as train
    job = os.environ['SLURM_JOB_ID']
    output = ROOT / 'outputs' / f'afrikaans-24h-{job}'
    if output.exists():
        raise RuntimeError(f'Output exists: {output}; use an explicit reviewed resume.')
    args = build_parser().parse_args([
        '--resume_training', '--resume_dir', str(ROOT / 'inputs/afrikaans-pilot'),
        '--output_dir', str(output), '--resume_to_step', '30500',
        '--max_train_seconds', '79200', '--validation_max_examples', '128',
        '--validation_shuffle', '--skip_expensive_metrics', '--fail_on_validation_error'])
    args = merge_saved_args(args)
    args.enc_dec_model = str(ROOT / 'models/mt5-base')
    args.dataset_dir = str(ROOT / 'data/nchlt_afrikaans/processed/stage_5_7')
    args.mixed_precision = 'bf16' if torch.cuda.is_bf16_supported() else 'no'
    args.fail_on_validation_error = True
    if args.lm_mode != 'decoder_last' or not args.backbone_eval_mode or args.overfit_subset:
        raise RuntimeError('Unexpected source checkpoint configuration')
    print('Precision:', args.mixed_precision, 'Output:', output, flush=True)
    print('Continuing optimizer and model from pilot step 500; target 30500 or 22-hour budget.', flush=True)
    train(args)


if __name__ == '__main__':
    main()
