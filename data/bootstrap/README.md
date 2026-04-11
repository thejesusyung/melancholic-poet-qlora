# Bootstrap Dataset

This directory contains a deliberately small first-pass dataset for a cheap Colab run.

- `bootstrap_train.jsonl`: 48 mixed training rows
- `bootstrap_val.jsonl`: 12 held-out validation rows
- `bootstrap_eval_prompts.jsonl`: 12 prompt specs for heuristic evaluation

Design goals:

- stronger poet rows than the current root `data/json*.json`
- explicit neutral counterbalance rows
- style-resistance rows that force format obedience
- safety rows that retain refusal and supportive behavior
- evaluation under both `poet` and `neutral` system prompts

This set is not meant to be final. It is meant to answer one question quickly: does a tiny, better-shaped dataset visibly move the adapter in the intended direction without obvious utility collapse?
