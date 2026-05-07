#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --config configs/default.yaml --dataset primekgqa --split test --checkpoint checkpoints/primekgqa.pt --out_dir results/primekgqa_test
