from __future__ import annotations

import argparse
from pathlib import Path

from poetune.evaluation import (
    aggregate_scores,
    load_eval_prompts,
    resolve_system_prompt,
    score_output,
    write_eval_artifacts,
)
from poetune.model_utils import cleanup_memory, default_generation_kwargs, generate_text, load_inference_model


def parse_adapter(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Adapter spec must look like name=path, got {raw}")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base model vs one or more adapters using heuristic metrics.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter spec: name=path")
    parser.add_argument("--prompt_file", type=str, default="data/sample/eval_prompts.jsonl")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    args = parser.parse_args()

    prompts = load_eval_prompts(args.prompt_file)
    variants = [("base", None)] + [parse_adapter(raw) for raw in args.adapter]
    generation_kwargs = default_generation_kwargs(for_eval=True)
    generation_kwargs.update(
        {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }
    )

    generations = []
    scored_rows = []

    for variant_name, adapter_path in variants:
        tokenizer, model, _ = load_inference_model(args.base_model, adapter_path=adapter_path)
        for prompt_spec in prompts:
            output = generate_text(
                tokenizer=tokenizer,
                model=model,
                base_model=args.base_model,
                prompt=prompt_spec["prompt"],
                system_prompt=resolve_system_prompt(prompt_spec),
                generation_kwargs=generation_kwargs,
            )
            generations.append(
                {
                    "variant": variant_name,
                    "prompt_id": prompt_spec["id"],
                    "category": prompt_spec["category"],
                    "system_prompt": resolve_system_prompt(prompt_spec),
                    "prompt": prompt_spec["prompt"],
                    "output": output,
                }
            )
            score_row = score_output(prompt_spec, output)
            score_row["variant"] = variant_name
            scored_rows.append(score_row)
        del model
        cleanup_memory()

    summary = aggregate_scores(scored_rows)
    write_eval_artifacts(args.out_dir, generations, scored_rows, summary)
    print(Path(args.out_dir, "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
