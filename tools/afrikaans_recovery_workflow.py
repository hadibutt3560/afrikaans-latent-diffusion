"""Continue the authorized recovery experiments with conservative validation gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from afrikaans_recovery import ROOT, OUT, command, run, summarize


def status(stage, **details):
    record = {'stage': stage, 'updated_unix': time.time(), **details}
    temporary = OUT / 'workflow-status.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(OUT / 'workflow-status.json')
    print(json.dumps(record), flush=True)


def metrics(directory):
    files = list(directory.glob('autoencoder-valid-step-*.json'))
    latest = max(files, key=lambda p: int(p.stem.rsplit('-', 1)[1]))
    return json.loads(latest.read_text())


def evaluate(name, checkpoint, examples):
    cmd = [sys.executable, '-u', 'train_latent_model.py', '--eval',
           '--resume_dir', str(checkpoint), '--output_dir', str(OUT / name),
           '--validation_shuffle', '--validation_max_examples', str(examples),
           '--skip_expensive_metrics']
    run(name, cmd)
    return metrics(OUT / name)


def main():
    os.chdir(ROOT)
    os.environ.update(WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
                      HF_HOME='/home/scientist/.cache/huggingface', PYTHONUNBUFFERED='1')
    status('waiting_for_diagnostics')
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        summary_file = OUT / 'summary.json'
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            if all(f'diagnostic-{mode}-250' in summary for mode in ('freeze', 'decoder_last')):
                break
        time.sleep(15)
    else:
        raise RuntimeError('Diagnostics did not finish within two hours; inspect console logs')

    baseline = OUT / 'baseline-step6000'
    status('baseline_validation_128')
    reference = evaluate('baseline-validation-128', baseline, 128)
    candidates = {}
    for mode in ('freeze', 'decoder_last'):
        name = f'pilot-{mode}-500'
        status(name)
        # Both candidates start from the original baseline, never the memorization runs.
        run(name, command(OUT / name, mode, 500, baseline))
        candidates[mode] = metrics(OUT / name)
        summarize()
    similarity = 'autoencoder/beam/character_similarity'
    ce = 'autoencoder/valid_cross_entropy'
    selected = max(candidates, key=lambda mode: candidates[mode][similarity])
    comparison = {'baseline': reference, 'candidates': candidates, 'selected': selected}
    (OUT / 'pilot-comparison.json').write_text(json.dumps(comparison, indent=2))
    # This is a compute-allocation gate, NOT a claim of diffusion readiness.
    if not (candidates[selected][ce] < reference[ce]
            and candidates[selected][similarity] > reference[similarity] + 0.02):
        status('capacity_comparison')
        for width in (64, 128):
            name = f'capacity-decoder_last-width{width}-500'
            run(name, command(OUT / name, 'decoder_last', 500, None, width=width))
        summarize()
        status('needs_review', reason='No clear pilot improvement; capacity comparison completed. No long run launched.')
        return

    status('long_training', selected=selected, additional_steps=3000)
    name = f'long-{selected}-3000'
    run(name, command(OUT / name, selected, 3000, OUT / f'pilot-{selected}-500' / 'best'))
    status('expanded_validation_512')
    evaluate('baseline-validation-512', baseline, 512)
    evaluate('candidate-validation-512', OUT / name / 'best', 512)
    status('trained_latent_ablation')
    with (OUT / 'latent-ablation.log').open('w') as log:
        subprocess.run([sys.executable, '-u', 'tools/inspect_autoencoder_latents.py',
                        '--checkpoint', str(OUT / name / 'best'),
                        '--output', str(OUT / 'latent-ablation.json')],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summarize()
    status('needs_quality_review', reason='Experiments complete. Inspect paired reconstructions before diffusion; test split was not scored.', selected=selected)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
