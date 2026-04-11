#!/usr/bin/env bash
set -euo pipefail

python generate_data.py \
  --out_dir data/generated \
  --persona_direct 60 \
  --counterbalance 60 \
  --logic 60 \
  --safety_confusing 60 \
  --val_total 32 \
  --seed 1337
