"""Read-only observation of predicted clean latents during sampling."""
from contextlib import contextmanager
import re


def word_tokens(text):
    return re.findall(r"\w+(?:['-]\w+)*|[^\w\s]", text, flags=re.UNICODE)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def annotate(rows):
    seen, previous = {}, None
    result = []
    for row in rows:
        text = row['text']
        words = word_tokens(text)
        result.append(dict(row, changed_from_previous=None if previous is None else text != previous['text'],
                           word_edits_from_previous=None if previous is None else edit_distance(word_tokens(previous['text']), words),
                           revisits_earlier_text=bool(previous is not None and text != previous['text'] and text in seen),
                           earlier_observed_step=seen.get(text), word_count=len(words)))
        seen.setdefault(text, row['step'])
        previous = row
    return result


@contextmanager
def capture_predictions(diffusion, every):
    if every < 1:
        raise ValueError('Observation interval must be positive')
    original = diffusion.diffusion_model_predictions
    prior_instance_attribute = diffusion.__dict__.get('diffusion_model_predictions')
    captured = []
    calls = 0

    def observed(*args, **kwargs):
        nonlocal calls
        prediction = original(*args, **kwargs)
        calls += 1
        if calls == 1 or calls % every == 0 or calls == diffusion.sampling_timesteps:
            # Copy only. Decode after sampling so observation cannot consume sampler RNG.
            captured.append((calls, prediction.pred_x_start.detach().float().cpu().clone()))
        return prediction

    diffusion.diffusion_model_predictions = observed
    try:
        yield captured
    finally:
        if prior_instance_attribute is None:
            delattr(diffusion, 'diffusion_model_predictions')
        else:
            diffusion.diffusion_model_predictions = prior_instance_attribute
