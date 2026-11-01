"""Summarize reconstruction errors without loading a model or changing data.

Input: a JSONL file of saved autoencoder reconstructions; each row needs a
"reference" (the original text) and a "beam_autoencoder" (the reconstruction
being audited) field.

Output: a JSON report at --output with the exact-match rate, mean character
similarity (difflib ratio), number recall, similarity by reference length and
the 20 least similar examples. The same summary, minus the examples, is printed.

Usage: python tools/audit_afrikaans_reconstructions.py EXAMPLES.jsonl --output REPORT.json
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re


# Digit runs joined by "." or ",", so "1,000", "3.5" and "1.234,56" each count
# as one number.
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')


def main():
    """Read the reconstruction examples, compute the error summary and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument('examples', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.examples.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('No reconstruction examples found')

    buckets = defaultdict(list)
    scored = []
    exact = 0
    expected_numbers = matched_numbers = 0
    changed_number_examples = 0
    for index, row in enumerate(rows):
        reference = row['reference']
        prediction = row['beam_autoencoder']
        # Character-level ratio 2*M/T in [0, 1]. autojunk=False turns off
        # difflib's heuristic that ignores very frequent characters once the
        # second string reaches 200 characters, which can distort long texts.
        similarity = SequenceMatcher(None, reference, prediction, autojunk=False).ratio()
        # Numbers are compared as multisets: order is ignored, counts are not.
        # The & intersection counts each reference number at most as often as it
        # appears in the prediction, giving number recall.
        ref_numbers, pred_numbers = Counter(NUMBER.findall(reference)), Counter(NUMBER.findall(prediction))
        expected_numbers += sum(ref_numbers.values())
        matched_numbers += sum((ref_numbers & pred_numbers).values())
        # Also counts examples where the prediction adds numbers.
        if ref_numbers != pred_numbers:
            changed_number_examples += 1
        exact += prediction == reference
        # Buckets use whitespace words of the reference; 50 is also the default
        # segment word cap in finalize_nchlt_afrikaans.py.
        words = len(reference.split())
        label = 'short_1_20' if words <= 20 else 'medium_21_50' if words <= 50 else 'long_51_plus'
        buckets[label].append(similarity)
        scored.append({
            'index': index, 'similarity': similarity, 'reference_words': words,
            'reference_numbers': list(ref_numbers.elements()),
            'prediction_numbers': list(pred_numbers.elements()),
            'reference': reference, 'prediction': prediction,
        })

    # Ascending, so the least similar examples come first (ties keep input order).
    scored.sort(key=lambda item: item['similarity'])
    report = {
        'examples': len(rows),
        'exact_reconstruction': exact / len(rows),
        'mean_character_similarity': sum(item['similarity'] for item in scored) / len(scored),
        # None when no reference contains a number.
        'number_recall': matched_numbers / expected_numbers if expected_numbers else None,
        'examples_with_changed_numbers_fraction': changed_number_examples / len(rows),
        'length_buckets': {
            name: {'count': len(values), 'mean_character_similarity': sum(values) / len(values)}
            for name, values in sorted(buckets.items())
        },
        'worst_20': scored[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'worst_20'}, indent=2))


if __name__ == '__main__':
    main()
