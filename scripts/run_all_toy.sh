#!/usr/bin/env bash
set -euo pipefail
python -m resemattn.data.toy_data --out_dir data/biohopr
python train.py --config configs/default.yaml --dataset biohopr --split train --output checkpoints/biohopr.pt
python evaluate.py --config configs/default.yaml --dataset biohopr --split test --checkpoint checkpoints/biohopr.pt --out_dir results/toy_biohopr
python visualize_answer_ranks.py --predictions results/toy_biohopr/predictions.jsonl --out results/toy_biohopr/rank_heatmap.png
