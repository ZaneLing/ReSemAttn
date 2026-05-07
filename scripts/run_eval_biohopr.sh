#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --config configs/default.yaml --dataset biohopr --split test --checkpoint checkpoints/biohopr.pt --out_dir results/biohopr_test
