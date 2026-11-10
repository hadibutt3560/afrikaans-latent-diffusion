"""Random-latent perplexity on the first 256 outputs, for the 256-sample validation column of the paper's Table 3.

Usage: python first256.py <outputs/latent-controls-YYYYMMDD> <outputs/ar-validation-audit-20261022>

Checks scored-C1.jsonl against its hash in results/SHA256SUMS.txt, pools mGPT token NLL over the first
256 C1 outputs (seeds 30000-30255), and divides by the AR primary arm's pooled perplexity on its 256
validation samples. Writes results/first256.json and never overwrites it.
"""
import sys, json, math, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMS = HERE / 'results/SHA256SUMS.txt'
OUT = HERE / 'results/first256.json'


def rows(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def pooled(rs):
    nll = sum(r['mgpt']['nll'] if 'mgpt' in r else r['nll'] for r in rs)
    tok = sum(r['mgpt']['tokens'] if 'mgpt' in r else r['tokens'] for r in rs)
    return nll, tok, math.exp(nll / tok)


def main(controls, audit):
    scored = Path(controls) / 'scored-C1.jsonl'
    expected = next(line.split()[0] for line in SUMS.read_text(encoding='utf8').splitlines()
                    if line.strip().endswith('scored-C1.jsonl'))
    actual = hashlib.sha256(scored.read_bytes()).hexdigest()
    if actual != expected:
        sys.exit(f'scored-C1.jsonl hash {actual} does not match SHA256SUMS {expected}')
    c1 = rows(scored)
    first = c1[:256]
    assert [r['index'] for r in first] == list(range(256)), 'C1 rows are not in seed order'
    nll, tok, ppl = pooled(first)
    _, _, ppl_all = pooled(c1)
    ar = rows(Path(audit) / 'ar-cap127-scored.jsonl')
    assert len(ar) == 256, len(ar)
    _, _, ppl_ar = pooled(ar)
    result = dict(
        source='scored-C1.jsonl', source_sha256=actual, n=256, seeds='30000-30255',
        nll_sum=nll, tokens=tok, perplexity=ppl, perplexity_all_1000=ppl_all,
        ar_source='ar-validation-audit-20261022/ar-cap127-scored.jsonl', ar_n=256, ar_perplexity=ppl_ar,
        ratio_to_ar=ppl / ppl_ar)
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
