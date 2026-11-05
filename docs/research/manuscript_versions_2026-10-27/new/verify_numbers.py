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
    'audit: numeral precision': f"Numeral precision & {numerals['matched']}/{numerals['produced']}",
    'audit: numeral precision rate': f"precision {numerals['precision']:.2f})",
    'audit: invented numerals': f"while {numerals['invented']} numerals appear",
    'corpus: shared 12-word spans': f"{split_overlap['passages_sharing_12gram']} of the {split_overlap['validation_passages']:,} validation passages",
    'shuffle: chrF range over five seeds': f"chrF {shuffle['chrf_range'][0]:.1f} to {shuffle['chrf_range'][1]:.1f}",
    'fresh: reconstruction': f"reconstructs only {fresh['exact']} exactly ({100 * fresh['exact_rate']:.1f}\\%), with chrF {fresh['chrf']:.1f}, "
                             f"numeral recall {fresh['numeral_recall']:.2f} and precision {fresh['numeral_precision']:.2f}",
    'random banks: raw perplexity': ', '.join(f'{x:.1f}' for x in iso[:-1]) + f' and {iso[-1]:.1f}',
}

# AR test split, original and retrained diffusion (ar_test_2026-11-05)
import statistics
art = load('ar_test_2026-11-05/results/analysis.json')
rt = art['ratios']
ci = lambda r: f"{r['ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
ci2 = lambda r: f"{r['ratio']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
run_ppl = lambda name: [v['ppl'] for v in art['systems'][name].values()]
msd = lambda name: f"{statistics.mean(run_ppl(name)):.2f} $\\pm$ {statistics.stdev(run_ppl(name)):.2f}"
msd_math = lambda name: f"${statistics.mean(run_ppl(name)):.2f}\\pm{statistics.stdev(run_ppl(name)):.2f}$"
CHECKS.update({
    'test: original small PPL': msd('B0-small'),
    'test: original large PPL': msd('B0-published'),
    'test: retrained large PPL': msd_math('D-published'),
    'test: retrained small PPL': msd_math('D-small'),
    'AR test: perplexity': f"score {art['banks']['AR']['ppl']:.2f} ({art['banks']['AR']['std_ppl']:.2f} length-standardised)",
    'AR test: training copies': f"{art['banks']['AR']['exact_copies']}/1,000",
    'AR test: original ratios': f"{ci(rt['B0-small ppl'])} and {ci(rt['B0-published ppl'])}, and {ci(rt['B0-small std_ppl'])} and {ci(rt['B0-published std_ppl'])}",
    'AR test: retrained large ratios': f"{ci2(rt['D-published ppl'])} raw and {ci2(rt['D-published std_ppl'])} length-standardised",
    'AR test: retrained small ratio': ci2(rt['D-small ppl']),
})

# Follow-up checks (followup_inference_2026-11-03)
fu = load('followup_inference_2026-11-03/analysis.json')
f2, f3, f1 = fu['f2_prior'], fu['f3_reconstruction'], fu['f1_audit']
c2 = sorted(f2[f'C2-D-{b}']['std_ppl'] for b in (40000, 41000, 42000))
large = sorted(f2[f'D-diffusion-published-seed{s}']['std_ppl'] for s in (42, 43, 44))
b0_large = sorted(-f1[f'B0-published-seed{s}']['paired_mean'] for s in (42, 43, 44))
d_small = sorted(-f1[f'D-small-seed{s}']['paired_mean'] for s in (42, 43, 44))
CHECKS.update({
    'follow-up: random latents on D': f"score {f2['D-isotropic']['ppl']:.1f} (standardised {f2['D-isotropic']['std_ppl']:.1f})",
    'follow-up: fitted prior on D': f"{c2[0]:.0f} to {c2[-1]:.0f} over three banks of 1,000, against {large[0]:.1f} to {large[-1]:.1f}",
    'follow-up: 681 exact counts': f"arms A and D reconstruct {f3['A']['all']['exact']} and {f3['D']['all']['exact']} exactly against {f3['B0']['all']['exact']}",
    'follow-up: 681 chrF': f"chrF {f3['A']['all']['chrf']:.1f} and {f3['D']['all']['chrf']:.1f} against {f3['B0']['all']['chrf']:.1f}",
    'follow-up: audit original large': f"by {b0_large[0]:.2f} to {b0_large[-1]:.2f} nats per token for the original large",
    'follow-up: audit retrained small': f"by {d_small[0]:.2f} to {d_small[-1]:.2f} for the retrained small",
    'follow-up: audit verdicts': 'inconclusive for the original denoisers and material for the retrained' if
        (fu['f1_decision']['B0']['verdict'], fu['f1_decision']['D']['verdict']) == ('inconclusive', 'material effect') else 'verdicts changed',
})

# Fertility (fertility_2026-11-05)
fe = load('fertility_2026-11-05/results/fertility.json')
gen = [v['mean'] if isinstance(v, dict) else v for k, v in fe['generation_banks'].items() if k != 'human test']
CHECKS.update({
    'fertility: generated banks': f"({min(gen):.2f} to {max(gen):.2f} tokens per word for both diffusion systems and AR, against {fe['generation_banks']['human test']:.2f})",
    'fertility: D exact in top third': f"arm D reconstructs {100 * fe['recon681_by_source_fertility']['D']['exact_by_tercile'][2]:.0f}\\% exactly",
})

# AR settings grid (evaluation_package_2026-10-31)
grid = {k: v for k, v in load('evaluation_package_2026-10-31/results/analysis.json')['banks'].items() if k.startswith('AR-grid')}
below = [v for v in grid.values() if v['mgpt']['standardised'] < 59.6]
CHECKS['AR grid: settings beating diffusion'] = (
    f"{['Seven', 'Eight'][len(below) - 7] if 7 <= len(below) <= 8 else len(below)} settings at temperature 0.8 or 0.9 nonetheless score below both diffusion sizes on length-standardised mGPT "
    f"({min(v['mgpt']['standardised'] for v in below):.1f} to {max(v['mgpt']['standardised'] for v in below):.1f}, against 59.6 and 71.3), "
    f"while copying {min(v['part_b']['training_copies'] for v in below)} to {max(v['part_b']['training_copies'] for v in below)}")

failed = [name for name, text in CHECKS.items() if text not in TEX]
for name, text in CHECKS.items():
    print(('FAIL ' if name in failed else 'ok   ') + name + ('' if name not in failed else f'  expected: {text}'))
sys.exit(1 if failed else 0)
