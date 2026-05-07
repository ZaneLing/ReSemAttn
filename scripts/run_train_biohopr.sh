#!/usr/bin/env bash
set -euo pipefail
python train.py --config configs/default.yaml --dataset biohopr --split train --output checkpoints/biohopr.pt
