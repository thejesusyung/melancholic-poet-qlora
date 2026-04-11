#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-Explain why eclipses do not happen every month.}"

python compare.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter persona=outputs/persona_only_qwen25_15b/adapter \
  --adapter mixed=outputs/mixed_qwen25_15b/adapter \
  --prompt "$PROMPT"
