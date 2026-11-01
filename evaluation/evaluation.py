from collections import defaultdict
import re
import warnings

import numpy as np
import torch
from evaluate import load
from nltk.util import ngrams


_METRIC_FAILURE_WARNINGS = set()
_STRICT_METRICS = False
_SKIP_EXPENSIVE_METRICS = False


def set_metric_policy(strict=False, skip_expensive=False):
    """Configure metric failure behaviour for a run.

    Training-time sampling should be non-strict. Explicit final evaluation must
    use strict=True so missing required metrics produce a non-zero exit.
    """
    global _STRICT_METRICS, _SKIP_EXPENSIVE_METRICS
    _STRICT_METRICS = bool(strict)
    _SKIP_EXPENSIVE_METRICS = bool(skip_expensive)


def _get_spacy_tokenizer():
    try:
        import spacy
        return spacy.load("en_core_web_sm").tokenizer
    except Exception:
        return None


def _tokenize(text, tokenizer=None):
    if tokenizer is not None:
        return [str(token) for token in tokenizer(text)]
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _metric_failure(metric_name, exc, strict=None, fallback=float("nan")):
    should_raise = _STRICT_METRICS if strict is None else strict
    if should_raise:
        raise RuntimeError(f"Required metric '{metric_name}' failed") from exc
    key = (metric_name, type(exc).__name__, str(exc))
    if key not in _METRIC_FAILURE_WARNINGS:
        _METRIC_FAILURE_WARNINGS.add(key)
        warnings.warn(f"{metric_name} metric unavailable: {exc}. Returning NaN.")
    return fallback


def _validate_parallel(predictions, references):
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions/references length mismatch: {len(predictions)} != {len(references)}"
        )


def compute_perplexity(all_texts_list, model_id="gpt2-large", strict=None):
    if not all_texts_list or not all(all_texts_list):
        return _metric_failure("perplexity", ValueError("empty generation"), strict)
    if _SKIP_EXPENSIVE_METRICS:
        return float("nan")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        perplexity = load("perplexity", module_type="metric")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = perplexity.compute(
            predictions=all_texts_list,
            model_id=model_id,
            device=device,
        )
        return float(results["mean_perplexity"])
    except Exception as exc:
        return _metric_failure("perplexity", exc, strict)


def compute_wordcount(all_texts_list):
    tokenizer = _get_spacy_tokenizer()
    words = {
        token.lower()
        for text in all_texts_list
        for token in _tokenize(text, tokenizer)
        if any(character.isalnum() for character in token)
    }
    return len(words)


def compute_unique_sentence_count(all_texts_list):
    return len({text.strip() for text in all_texts_list if text.strip()})


def compute_diversity(all_texts_list):
    tokenizer = _get_spacy_tokenizer()
    token_list = [_tokenize(sentence, tokenizer) for sentence in all_texts_list]
    metrics = {}
    diversity = 1.0
    for n in [2, 3, 4]:
        unique_ngrams = set()
        count = 0
        for tokens in token_list:
            sentence_ngrams = list(ngrams(tokens, n))
            unique_ngrams.update(sentence_ngrams)
            count += len(sentence_ngrams)
        distinct = len(unique_ngrams) / count if count else 0.0
        repetition = 1.0 - distinct if count else 0.0
        metrics[f"distinct_{n}"] = distinct
        metrics[f"{n}gram_repetition"] = repetition
        # Preserve the misspelled upstream key for old dashboards.
        metrics[f"{n}gram_repitition"] = repetition
        diversity *= distinct
    metrics["diversity"] = diversity
    return metrics


def compute_memorization(all_texts_list, human_references, n=4):
    tokenizer = _get_spacy_tokenizer()
    reference_ngrams = set()
    for sentence in human_references:
        reference_ngrams.update(ngrams(_tokenize(sentence, tokenizer), n))

    generated_ngrams = [
        ngram
        for sentence in all_texts_list
        for ngram in ngrams(_tokenize(sentence, tokenizer), n)
    ]
    if not generated_ngrams:
        return 0.0
    duplicates = sum(ngram in reference_ngrams for ngram in generated_ngrams)
    return duplicates / len(generated_ngrams)


def compute_mauve(all_texts_list, human_references, model_id, strict=None):
    _validate_parallel(all_texts_list, human_references)
    if not all_texts_list:
        return _metric_failure("mauve", ValueError("empty generation"), strict), []
    if _SKIP_EXPENSIVE_METRICS:
        return float("nan"), []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        if model_id != "gpt2-large":
            raise ValueError("The paper-reproduction MAUVE model must be gpt2-large")
        mauve = load("mauve")
        results = mauve.compute(
            predictions=all_texts_list,
            references=human_references,
            featurize_model_name=model_id,
            max_text_length=256,
            device_id=0 if torch.cuda.is_available() else -1,
        )
        return float(results.mauve), results.divergence_curve
    except Exception as exc:
        return _metric_failure("mauve", exc, strict), []


def compute_bleu(all_texts_list, human_references):
    _validate_parallel(all_texts_list, human_references)
    if not all_texts_list:
        return 0.0
    bleu = load("bleu")
    results = bleu.compute(
        predictions=all_texts_list,
        references=[[ref] for ref in human_references],
    )
    return float(results["bleu"])


def compute_sacrebleu(all_texts_list, human_references, tokenize, use_effective_order=False):
    _validate_parallel(all_texts_list, human_references)
    if not all_texts_list:
        return 0.0
    sacrebleu = load("sacrebleu")
    results = sacrebleu.compute(
        predictions=all_texts_list,
        references=[[ref] for ref in human_references],
        tokenize=tokenize,
        use_effective_order=use_effective_order,
    )
    return float(results["score"])


def compute_debertascore(all_texts_list, human_references):
    _validate_parallel(all_texts_list, human_references)
    bert = load("bertscore")
    results = bert.compute(
        predictions=all_texts_list,
        references=[[ref] for ref in human_references],
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
    )
    del results["hashcode"]
    return {key: np.asarray(value).mean() for key, value in results.items()}


def compute_bertscore(all_texts_list, human_references):
    _validate_parallel(all_texts_list, human_references)
    bert = load("bertscore")
    results = bert.compute(
        predictions=all_texts_list,
        references=[[ref] for ref in human_references],
        lang="en",
        rescale_with_baseline=True,
    )
    del results["hashcode"]
    return {key: np.asarray(value).mean() for key, value in results.items()}


def compute_rouge(all_texts_list, human_references, use_aggregator=True, use_stemmer=False):
    _validate_parallel(all_texts_list, human_references)
    if not all_texts_list:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    rouge = load("rouge")
    return rouge.compute(
        predictions=all_texts_list,
        references=human_references,
        use_aggregator=use_aggregator,
        use_stemmer=use_stemmer,
    )
