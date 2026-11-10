"""Check trained latents and compare correct, shuffled and zero latent conditioning."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers.modeling_outputs import BaseModelOutput
from latent_models.latent_utils import get_latent_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--dataset-dir', type=Path, default=None)
    parser.add_argument('--split', choices=('train', 'valid', 'test'), default='valid')
    parser.add_argument('--examples', type=int, default=16)
    args = parser.parse_args()
    config = SimpleNamespace(**json.loads((args.checkpoint / 'args.json').read_text()))
    if args.model_name:
        config.enc_dec_model = args.model_name
    model, tokenizer, _ = get_latent_model(config)
    checkpoint = torch.load(args.checkpoint / 'model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model = model.cuda().eval()
    dataset = args.dataset_dir or ROOT / 'data/nchlt_afrikaans/processed/stage_5_7'
    rows = [json.loads(line)['text'] for line in (dataset / f'{args.split}.jsonl').open(encoding='utf-8')]
    import random
    random.Random(42).shuffle(rows)
    rows = rows[:args.examples]
    losses = {name: 0.0 for name in ('correct', 'shuffled', 'zero')}
    count = 0
    latents = []
    for index in range(0, len(rows), 2):
        batch = tokenizer(rows[index:index + 2], return_tensors='pt', padding=True,
                          truncation=True, max_length=config.max_seq_len).to('cuda')
        labels = batch.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokens = int((labels != -100).sum())
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else torch.autocast('cuda', enabled=False)
        with autocast:
            encoded = model.get_encoder()(**batch)
            z = model.get_diffusion_latent(encoded, batch.attention_mask)
            latents.append(z.float().cpu())
            for name, latent in [('correct', z), ('shuffled', z.flip(0)), ('zero', torch.zeros_like(z))]:
                decoded = BaseModelOutput(last_hidden_state=model.get_decoder_input(latent))
                loss = model(encoder_outputs=decoded, labels=labels).loss
                losses[name] += float(loss) * tokens
        count += tokens
    z = torch.cat(latents)
    report = {'examples': len(rows), 'latent_shape': list(z.shape),
              'all_finite': bool(torch.isfinite(z).all()),
              'mean_coordinate_std_across_examples': float(z.std(dim=0).mean()),
              'mean_pairwise_distance': float(torch.pdist(z.flatten(1)).mean()),
              'conditional_cross_entropy': {name: value / count for name, value in losses.items()},
              'interpretation': 'Correct conditioning should outperform shuffled/zero conditioning. This small diagnostic is not a diffusion-readiness certificate.'}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
