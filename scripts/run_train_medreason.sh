#!/usr/bin/env bash
set -euo pipefail
python train.py --config configs/default.yaml --dataset medreason --split train --output checkpoints/medreason.pt
