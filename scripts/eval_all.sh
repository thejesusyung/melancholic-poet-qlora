#!/usr/bin/env bash
set -euo pipefail

python eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter persona=outputs/persona_only_qwen25_15b/adapter \
  --adapter mixed=outputs/mixed_qwen25_15b/adapter \
  --prompt_file data/sample/eval_prompts.jsonl \
  --out_dir reports/qwen25_15b_eval
