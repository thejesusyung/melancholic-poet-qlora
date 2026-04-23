from __future__ import annotations

import argparse
import json
from pathlib import Path

from poetune.constants import DEFAULT_NEUTRAL_SYSTEM_PROMPT, DEFAULT_POET_SYSTEM_PROMPT
from poetune.model_utils import cleanup_memory, default_generation_kwargs, generate_text, load_inference_model


def parse_adapter(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Adapter spec must look like name=path, got {raw}")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one prompt across the base model and one or more adapters.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter spec: name=path")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default="poet")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        parser.error("Provide --prompt or --prompt_file.")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    if args.system_prompt == "poet":
        system_prompt = DEFAULT_POET_SYSTEM_PROMPT
    elif args.system_prompt == "neutral":
        system_prompt = DEFAULT_NEUTRAL_SYSTEM_PROMPT
    else:
        system_prompt = args.system_prompt

    variants = [("base", None)] + [parse_adapter(raw) for raw in args.adapter]
    generation_kwargs = default_generation_kwargs(for_eval=False)
    generation_kwargs.update(
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        }
    )

    outputs = []
    for name, adapter_path in variants:
        tokenizer, model, device = load_inference_model(args.base_model, adapter_path=adapter_path)
        response = generate_text(
            tokenizer=tokenizer,
            model=model,
            base_model=args.base_model,
            prompt=prompt,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
        )
        outputs.append(
            {
                "variant": name,
                "adapter_path": adapter_path,
                "device": device,
                "response": response,
            }
        )
        del model
        cleanup_memory()

    if args.out_path:
        Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    for row in outputs:
        print("=" * 88)
        print(row["variant"])
        print("-" * 88)
        print(row["response"])
        print()


if __name__ == "__main__":
    main()
