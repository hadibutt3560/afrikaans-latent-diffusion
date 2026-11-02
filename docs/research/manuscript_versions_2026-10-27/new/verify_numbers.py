"""Check that numbers quoted in main.tex match the committed result files.

Usage: python verify_numbers.py   (CPU only; reads committed files, writes nothing)
Each check formats a value from a result file and requires that exact string in main.tex.
Extend CHECKS when a new result enters the manuscript.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
TEX = (HERE / 'main.tex').read_text(encoding='utf8')


def load(path):
    return json.loads((DOCS / path).read_text(encoding='utf8'))


screen = load('ae_screen_2026-10-27/screen_report_from_cluster.json')
swahili = load('swahili_check_2026-11-01/results/summary.json')
pct = lambda x: f'{100 * x:.1f}'
a_b0 = screen['V_contrasts']['A_minus_B0']['exact']
reserve = screen['reserve_contrast_winner_minus_B0']['exact']
winner, b0 = screen['winner_reserve'], screen['b0']['reserve']
recon = swahili['s1_reconstruction']
s4 = swahili['s4_length_and_copying']
diffusion = [k for k in s4 if k.endswith('-valid') and k != 'real-valid']
overlap = sorted(round(100 * s4[k]['fourgram_overlap']) for k in diffusion)
F = 'local_followups_2026-11-01/results/'
fresh, shuffle, banks = load(F + 'extra_monitor_summary.json'), load(F + 'shuffle_seeds.json'), load(F + 'random_banks.json')
numerals, split_overlap = load(F + 'numerals.json')['audit_81'], load(F + 'overlap.json')
iso = sorted(v['raw_ppl'] for k, v in banks.items() if k.startswith('C1') and 'spread' not in k)
std = lambda kind: banks[f'{kind}_spread']['standardised_range']

CHECKS = {
    'screen: recipe gain (A - B0, exact)': f"{pct(a_b0['difference'])} points (95\\% CI [{pct(a_b0['low'])}, {pct(a_b0['high'])}])",
    'screen: reserve exact counts': f"{winner['exact']} of {winner['n']} exact against {b0['exact']}",
    'screen: reserve exact difference': f"{pct(reserve['difference'])} points, CI [{pct(reserve['low'])}, {pct(reserve['high'])}]",
    'screen: reserve chrF': f"chrF {winner['chrf']:.2f} against {b0['chrf']:.2f}",
    'screen: reserve numeral recall': f"numeral recall {winner['number_recall']:.2f} against {b0['number_recall']:.2f}",
    'screen: long reserve passages': f"({winner['bins']['long']['exact']} of {winner['bins']['long']['n']})",
    'screen: long reserve chrF': f"chrF is {winner['bins']['long']['chrf']:.1f}",
    'swahili: chrF and interval': f"mean chrF {recon['correct']['chrf']:.1f} (95\\% CI [{recon['bootstrap_95']['chrf'][0]:.1f}, {recon['bootstrap_95']['chrf'][1]:.1f}])",
    'swahili: shuffled chrF': f"against {recon['shuffled']['chrf']:.1f} with shuffled latents",
    'swahili: no exact matches': 'no exact matches' if recon['correct']['exact'] == 0 else f"{recon['correct']['exact']} exact",
    'swahili: numeral recall': f"numeral recall was {recon['correct']['numeral_recall']:.2f}",
    'swahili: random-latent perplexity': f"(perplexity {swahili['s2_random_latents']['ppl']:.0f})",
    'swahili: random-latent Swahili share': f"against {100 * swahili['s3_language']['random-latents']['pooled']['swh_Latn']:.0f}\\% of random-latent outputs",
    'swahili: diffusion 4-gram overlap': f"{overlap[0]}--{overlap[-1]}\\% of their word 4-grams",
    'swahili: reference 4-gram overlap': f"against {100 * s4['real-valid']['fourgram_overlap']:.0f}\\% for references",
    'audit: numeral precision': f"{numerals['matched']}/{numerals['produced']} ({numerals['precision']:.2f})",
    'audit: invented numerals': f"while {numerals['invented']} numerals appear",
    'corpus: shared 12-word spans': f"{split_overlap['passages_sharing_12gram']} of the {split_overlap['validation_passages']:,} validation passages",
    'shuffle: chrF range over five seeds': f"chrF {shuffle['chrf_range'][0]:.1f} to {shuffle['chrf_range'][1]:.1f}",
    'fresh: reconstruction': f"reconstructs only {fresh['exact']} exactly ({100 * fresh['exact_rate']:.1f}\\%), with chrF {fresh['chrf']:.1f}, "
                             f"numeral recall {fresh['numeral_recall']:.2f} and precision {fresh['numeral_precision']:.2f}",
    'random banks: raw perplexity': ', '.join(f'{x:.1f}' for x in iso[:-1]) + f' and {iso[-1]:.1f}',
    'random banks: standardised ranges': f"fitted-Gaussian banks score {std('C2')[0]:.1f} to {std('C2')[1]:.1f} and isotropic banks {std('C1')[0]:.1f} to {std('C1')[1]:.1f}",
}

failed = [name for name, text in CHECKS.items() if text not in TEX]
for name, text in CHECKS.items():
    print(('FAIL ' if name in failed else 'ok   ') + name + ('' if name not in failed else f'  expected: {text}'))
sys.exit(1 if failed else 0)
