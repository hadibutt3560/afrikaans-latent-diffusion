"""Reconstruction evaluation of one autoencoder on one fixed panel (inference only).

Per passage: teacher-forced token CE and accuracy, then deterministic beam decoding with the
benchmark settings, scored by exact match (against the tokenised model view), sentence chrF
(SacreBLEU 2.6.0 settings of the 22 October audit) and numeral precision/recall. FP32, batch one,
the same procedure as docs/research/train_heldout_reconstruction_2026-10-27/run_diagnostic.py.
"""
import argparse, json, os, re, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('BUNDLE_ROOT', HERE.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))  # sacrebleu is not installed in the cluster environment
BEAM = dict(max_length=128, min_length=5, num_beams=4, do_sample=False, no_repeat_ngram_size=3,
            repetition_penalty=1.2, length_penalty=1.0)
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def evaluate(ae_dir, model_cache, panel_path, output, decode=True):
    import torch
    from sacrebleu.metrics import CHRF
    from transformers.modeling_outputs import BaseModelOutput
    from tools.longrun_training import load_ae
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = json.loads((Path(ae_dir) / 'args.json').read_text(encoding='utf8'))
    ae, tok = load_ae(ae_dir, model_cache, dict(latent_dim=64, latent_length=args['num_encoder_latents'], max_length=128))
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    rows, start = [], time.monotonic()
    output = Path(output)
    with torch.inference_mode():
        for r in read_jsonl(panel_path):
            ids = tok(r['text'], truncation=True, max_length=128)['input_ids']
            view = tok.decode(ids, skip_special_tokens=True)
            x = torch.tensor([ids], device='cuda')
            mask = torch.ones_like(x)
            z = ae.get_diffusion_latent(ae.get_encoder()(input_ids=x, attention_mask=mask), mask)
            if not torch.isfinite(z).all():
                raise FloatingPointError(f"Non-finite latent for {r['id']}")
            enc = BaseModelOutput(last_hidden_state=ae.get_decoder_input(z))
            out = ae(encoder_outputs=enc, labels=x)
            ce = float(out.loss)
            correct = int((out.logits.argmax(-1) == x).sum())
            row = dict(id=r['id'], document_id=r['document_id'], bin=r.get('bin'), tokens=len(ids), ce=ce, nll=ce * len(ids),
                       correct_tokens=correct, latent_norm_mean=float(z.norm(dim=-1).mean()))
            if decode:
                gen = ae.generate(encoder_outputs=enc, **BEAM)[0].tolist()
                pred = tok.decode(gen, skip_special_tokens=True).strip()
                a, b = Counter(NUMBER.findall(view)), Counter(NUMBER.findall(pred))
                row.update(text=pred, source=r['text'], model_view=view, exact=int(pred == view),
                           chrf=chrf.sentence_score(pred, [view]).score, eos=tok.eos_token_id in gen, at_cap=len(gen) >= 128,
                           number_expected=sum(a.values()), number_predicted=sum(b.values()), number_matched=sum((a & b).values()))
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = summarize(rows)
    summary.update(seconds=time.monotonic() - start, ae_dir=str(ae_dir), panel=Path(panel_path).name)
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    del ae
    torch.cuda.empty_cache()
    return summary


def summarize(rows):
    tokens = sum(r['tokens'] for r in rows)
    s = dict(n=len(rows), ce=sum(r['nll'] for r in rows) / tokens, accuracy=sum(r['correct_tokens'] for r in rows) / tokens)
    if rows and 'exact' in rows[0]:
        expected = sum(r['number_expected'] for r in rows)
        predicted = sum(r['number_predicted'] for r in rows)
        matched = sum(r['number_matched'] for r in rows)
        s.update(exact=sum(r['exact'] for r in rows), exact_rate=sum(r['exact'] for r in rows) / len(rows),
                 chrf=sum(r['chrf'] for r in rows) / len(rows), number_recall=matched / expected if expected else None,
                 number_precision=matched / predicted if predicted else None, cap=sum(r['at_cap'] for r in rows),
                 no_eos=sum(not r['eos'] for r in rows))
        by_bin = {}
        for b in sorted({r['bin'] for r in rows if r.get('bin')}):
            rs = [r for r in rows if r['bin'] == b]
            by_bin[b] = dict(n=len(rs), exact=sum(r['exact'] for r in rs), chrf=sum(r['chrf'] for r in rs) / len(rs),
                             ce=sum(r['nll'] for r in rs) / sum(r['tokens'] for r in rs))
        s['bins'] = by_bin
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ae', type=Path, required=True)
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-decode', action='store_true')
    a = p.parse_args()
    print(json.dumps(evaluate(a.ae, a.model_cache, a.panel, a.output, not a.no_decode), indent=2))
