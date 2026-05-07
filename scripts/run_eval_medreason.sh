#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --config configs/default.yaml --dataset medreason --split test --checkpoint checkpoints/medreason.pt --out_dir results/medreason_test
