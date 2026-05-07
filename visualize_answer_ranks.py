"""Plot answer rank distribution / heat map using matplotlib.

Usage:
  python visualize_answer_ranks.py --predictions results/biohopr_test/predictions.jsonl --out results/rank_heatmap.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def first_gold_rank(row):
    ranked = sorted(row['candidates'], key=lambda x: x.get('score',0.0), reverse=True)
    for i, c in enumerate(ranked, start=1):
        if c.get('label',0) == 1:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', default='rank_heatmap.png')
    ap.add_argument('--max_rank', type=int, default=64)
    args = ap.parse_args()
    ranks = []
    for row in read_jsonl(args.predictions):
        r = first_gold_rank(row)
        if r is not None:
            ranks.append(min(r, args.max_rank))
    if not ranks:
        raise SystemExit('No gold candidates found in predictions')
    bins = np.arange(1, args.max_rank + 2)
    hist, _ = np.histogram(ranks, bins=bins)
    heat = hist.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(12, 2.2))
    im = ax.imshow(heat, aspect='auto')
    ax.set_yticks([0]); ax.set_yticklabels(['Gold answer density'])
    ax.set_xticks([0, 4, 9, 19, 31, 63])
    ax.set_xticklabels(['1', '5', '10', '20', '32', '64'])
    ax.set_xlabel('Rank position of first gold answer')
    ax.set_title('Distribution of First Correct Answer Position')
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02, label='count')
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f'saved {args.out}; n={len(ranks)}, mean_rank={np.mean(ranks):.2f}, median_rank={np.median(ranks):.2f}')

if __name__ == '__main__':
    main()
