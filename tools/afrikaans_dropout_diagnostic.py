"""Matched dropout ablation against the completed 250-step memorization control."""
import json
import os
import time

from afrikaans_recovery import OUT, command, run


def status(stage, **details):
    path = OUT / 'dropout-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'stage': stage, 'updated_unix': time.time(), **details}, indent=2))
    temporary.replace(path)


def main():
    os.environ.update(WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
                      HF_HOME='/home/scientist/.cache/huggingface', PYTHONUNBUFFERED='1')
    name = 'diagnostic-decoder_last-no-backbone-dropout-250'
    cmd = command(OUT / name, 'decoder_last', 250, OUT / 'baseline-step6000', tiny=True)
    cmd += ['--backbone_eval_mode']
    status('training', experiment=name)
    run(name, cmd)
    control = json.loads((OUT / 'diagnostic-decoder_last-250' / 'autoencoder-valid-step-250.json').read_text())
    candidate = json.loads((OUT / name / 'autoencoder-valid-step-250.json').read_text())
    comparison = {'control_dropout_on': control, 'candidate_dropout_off': candidate,
                  'scope': 'Same 32 training examples, initialization and 250 updates; memorization diagnostic, not held-out quality.',
                  'next': 'Review paired reconstructions before choosing any full-corpus follow-up.'}
    (OUT / 'dropout-comparison.json').write_text(json.dumps(comparison, indent=2))
    status('completed_needs_review', experiment=name)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
