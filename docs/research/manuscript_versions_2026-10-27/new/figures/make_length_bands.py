"""Length-band mGPT perplexity figure (test split, human and diffusion only).

Reads docs/research/stage4_final_evaluation_2026-10-23/length_bands.json and
writes figures/length_bands.pdf next to this script. No model is run.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
data = json.loads((ROOT / 'docs/research/stage4_final_evaluation_2026-10-23/length_bands.json').read_text(encoding='utf8'))
BANDS = ['1-10', '11-20', '21-35', '36+']
LABELS = ['1–10', '11–20', '21–35', '36+']

plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.5 / 2.54, 5.5 / 2.54))
x = range(len(BANDS))

human = data['test-reference-mgpt']
hy = [human[b]['ppl'] for b in BANDS]
ax.plot(x, hy, color='black', marker='o', ms=3.5, lw=1.2, label='Human text (1,000)')

for fam, colour, marker, name in (('small', 'tab:blue', '^', 'Small diffusion'), ('published', 'tab:purple', 'D', 'Large diffusion')):
    runs = [data[f'{fam}-seed{s}'] for s in range(42, 47)]
    vals = [[r[b]['ppl'] for r in runs] for b in BANDS]
    mean = [sum(v) / len(v) for v in vals]
    ax.fill_between(x, [min(v) for v in vals], [max(v) for v in vals], color=colour, alpha=0.2, lw=0)
    ax.plot(x, mean, color=colour, marker=marker, ms=3, lw=1.2, label=f'{name} (5 × 1,000)')

ax.set_yscale('log')
ax.set_yticks([20, 50, 100, 200, 500])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.minorticks_off()
ax.set_xticks(list(x))
ax.set_xticklabels([f"{l}\nn={human[b]['n']}" for l, b in zip(LABELS, BANDS)])
ax.set_xlabel('Output length (whitespace words)')
ax.set_ylabel('mGPT perplexity')
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(pad=0.3)
fig.savefig(HERE / 'length_bands.pdf')
print('written', HERE / 'length_bands.pdf')
