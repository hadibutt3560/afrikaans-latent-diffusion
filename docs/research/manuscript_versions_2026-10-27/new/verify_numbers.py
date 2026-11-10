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
    'fertility: D exact in top third': f"(arm D, \\S\\ref{{sec:res-aescreen}}) reconstructs {100 * fe['recon681_by_source_fertility']['D']['exact_by_tercile'][2]:.0f}\\% exactly",
})

# AR settings grid (evaluation_package_2026-10-31)
grid = {k: v for k, v in load('evaluation_package_2026-10-31/results/analysis.json')['banks'].items() if k.startswith('AR-grid')}
below = [v for v in grid.values() if v['mgpt']['standardised'] < 59.6]
CHECKS['AR grid: settings beating diffusion'] = (
    f"{['Seven', 'Eight'][len(below) - 7] if 7 <= len(below) <= 8 else len(below)} settings at temperature 0.8 or 0.9 nonetheless score below both diffusion sizes on length-standardised mGPT "
    f"({min(v['mgpt']['standardised'] for v in below):.1f} to {max(v['mgpt']['standardised'] for v in below):.1f}, against 59.6 and 71.3), "
    f"while copying {min(v['part_b']['training_copies'] for v in below)} to {max(v['part_b']['training_copies'] for v in below)}")

# Diversity and repetition (stage4_final_evaluation_2026-10-23)
import csv
s4 = load('stage4_final_evaluation_2026-10-23/analysis.json')['test_summary']
vrows = list(csv.DictReader((DOCS / 'stage4_final_evaluation_2026-10-23/validation_comparison.csv').open(encoding='utf8')))
vd1 = sorted(float(r['distinct_1']) for r in vrows if r['run'] != 'validation/AR')
var = next(r for r in vrows if r['run'] == 'validation/AR')
CHECKS.update({
    'diversity: test distinct-1': f"Test distinct-1 is {s4['small']['distinct_1']['mean']:.3f} (small) and {s4['published']['distinct_1']['mean']:.3f} (large)",
    'diversity: test repeated 3-grams': f"{100 * s4['small']['mean_repeat_3']['mean']:.3f}\\% and {100 * s4['published']['mean_repeat_3']['mean']:.3f}\\% of an output",
    'diversity: test repeated 4-grams': f"(4-grams: {100 * s4['small']['mean_repeat_4']['mean']:.0f} and {100 * s4['published']['mean_repeat_4']['mean']:.4f}\\%)",
    'diversity: validation distinct-1 and AR': f"diffusion distinct-1 is {vd1[0]:.2f} to {vd1[-1]:.2f} per run against {float(var['distinct_1']):.3f} for AR, whose outputs repeat {100 * float(var['mean_repeat_3']):.2f}\\% of their 3-grams",
})

# Claude content labels (claude_labels_2026-11-07)
cl = load('claude_labels_2026-11-07/agreement.json')
CHECKS.update({
    'claude labels: pass count': f"All content checks (Claude) & {cl['claude_counts']['pass']}/81",
    'claude labels: fail count': f"Claude marked {cl['claude_counts']['fail']} of 81 passages ({100 * cl['claude_counts']['fail'] / 81:.0f}\\%)",
    'claude labels: agreement': f"on {cl['agreement']['result']['agree']} of 81 (Cohen's $\\kappa={cl['agreement']['result']['kappa']:.2f}$, agreement corrected for chance)",
})

# Decoder learning curves (jobs 74177 and 75003, research workspace outputs)
import glob as _glob
_CR = DOCS.parents[2] / 'research/outputs/cluster-results'
def _curve(job, arm):
    out = []
    for f in sorted(_glob.glob(str(_CR / f'{job}/development-{arm}-*.json'))):
        step = int(Path(f).stem.split('-')[-1])
        out.append((step, json.loads(Path(f).read_text(encoding='utf-8-sig'))['metrics']['chrf']))
    return out
if _CR.exists():
    f12 = _curve('job-74177-review/job-74177', 'F12')
    d0 = _curve('job-75003-review/job-75003', 'D0')
    d1 = _curve('job-75003-review/job-75003', 'D1')
    later = [c for s_, c in f12 if s_ > f12[0][0]]
    gap = max(abs(a[1] - b[1]) for a, b in zip(f12, d0))
    CHECKS.update({
        'curve: first check': f"within {f12[0][0]:,} updates (chrF {f12[0][1]:.2f})",
        'curve: later range': f"later checks to {f12[-1][0]:,} stayed between {min(later):.2f} and {max(later):.2f}",
        'curve: same-seed rerun gap': f"tracked the earlier curve within {gap:.2f}",
        'curve: dropout range': f"With dropout, chrF stayed between {min(c for _, c in d1):.2f} and {max(c for _, c in d1):.2f}",
    })
else:
    print('skip: cluster results not found, decoder curve checks not run')

# Five review fixes (10 November 2026)
import re as _re
_fe = (DOCS / 'stage4_final_evaluation_2026-10-23/FINAL_EVALUATION.md').read_text(encoding='utf8')
_ce = _re.search(r'Reconstruction CE \| ([0-9.]+) \| ([0-9.]+)', _fe)
_ex = _re.search(r'Exact reconstruction \| ([0-9.]+)% \| ([0-9.]+)%', _fe)
_ar = (DOCS / 'ar_validation_audit_2026-10-22/AR_SUITABILITY_REPORT.md').read_text(encoding='utf8')
_hu = _re.search(r'Real-validation reference \| [^|]*\| 256 total \| ([0-9.]+)', _ar)
_sm = load('ar_test_2026-11-05/results/analysis.json')['system_means']
_r256 = load('latent_controls_2026-10-30/results/first256.json')
CHECKS.update({
    'recon: test CE and exact': f"reconstruction CE is {float(_ce.group(2)):.3f} and {float(_ex.group(2)):.1f}\\% are exact (validation: {float(_ce.group(1)):.3f} and {float(_ex.group(1)):.1f}\\%)",
    'table3: human 256': f"Human text & {float(_hu.group(1)):.2f} & --",
    'table3: random 256 ppl and ratio': f"Random latents & {_r256['perplexity']:.2f} & {_r256['ratio_to_ar']:.4f}",
    'retrained: standardised original': f"against {_sm['B0-published']['std_ppl']:.1f} and {_sm['B0-small']['std_ppl']:.1f};",
})

# Split fertility, same definition as the generated banks (fertility_2026-11-05/split_fertility.py)
_sf = load('fertility_2026-11-05/results/split_fertility.json')
CHECKS['fertility: validation split'] = f"validation fertility is {_sf['valid']['fertility']:.2f} mT5 tokens per word"

failed = [name for name, text in CHECKS.items() if text not in TEX]
for name, text in CHECKS.items():
    print(('FAIL ' if name in failed else 'ok   ') + name + ('' if name not in failed else f'  expected: {text}'))
sys.exit(1 if failed else 0)
