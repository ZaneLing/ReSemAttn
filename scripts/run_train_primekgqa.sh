#!/usr/bin/env bash
set -euo pipefail
python train.py --config configs/default.yaml --dataset primekgqa --split train --output checkpoints/primekgqa.pt
