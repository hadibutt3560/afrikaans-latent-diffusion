"""Summarize a controlled cycle without declaring automatic language success."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    for name in ('summary.json','generation_metrics.json'):
        path=args.run/name
        if path.exists():
            print(path)
            print(json.dumps(json.loads(path.read_text()),indent=2,ensure_ascii=False))
        else:
            print(f'{name}: unavailable; evaluation not confirmed complete')
    print('Human review: blind_review.csv. Keep review_key.csv hidden until ratings are locked.')
    print('No automatic success claim. Check matched training updates before comparing language.')


if __name__=='__main__':
    main()
