"""Evaluate the downloaded cluster checkpoint on local held-out data."""
import argparse
import os
from pathlib import Path
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--max-examples', type=int, default=1024)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    cli = parser.parse_args()
    if cli.max_examples <= 0 or cli.eval_batch_size <= 0:
        raise ValueError('example count and batch size must be positive')
    os.chdir(ROOT)
    os.environ.update(
        WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
        HF_HOME=str(Path.home() / '.cache/huggingface'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
    )
    os.environ.pop('HF_DATASETS_CACHE', None)
    os.environ.pop('TRANSFORMERS_CACHE', None)
    import torch
    from train_latent_model import build_parser, merge_saved_args, main as evaluate

    checkpoint = ROOT / 'outputs/cluster-results/afrikaans-56531/best'
    for name in ('model.pt', 'args.json'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA GPU in WSL; evaluation not started.')
    torch.ones(16, device='cuda').square().sum().item()
    output = ROOT / 'outputs/cluster-results' / datetime.now().strftime(
        f'afrikaans-{cli.split}{cli.max_examples}-%Y%m%d-%H%M%S-%f'
    )
    args = merge_saved_args(build_parser().parse_args([
        '--eval_test' if cli.split == 'test' else '--eval',
        '--resume_dir', str(checkpoint), '--output_dir', str(output),
        '--validation_max_examples', str(cli.max_examples), '--validation_shuffle',
        '--skip_expensive_metrics',
    ]))
    args.enc_dec_model = 'google/mt5-base'
    args.dataset_dir = str(ROOT / 'data/nchlt_afrikaans/processed/stage_5_7')
    args.mixed_precision = 'bf16' if torch.cuda.is_bf16_supported() else 'no'
    args.eval_batch_size = cli.eval_batch_size
    args.dataloader_num_workers = 0
    if args.overfit_subset or args.dataset_name != 'nchlt_afrikaans':
        raise RuntimeError('Unexpected checkpoint dataset settings')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Torch:', torch.__version__, 'Precision:', args.mixed_precision, flush=True)
    print('Evaluation output:', output, flush=True)
    evaluate(args)


if __name__ == '__main__':
    main()
