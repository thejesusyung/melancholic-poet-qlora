from __future__ import annotations

import argparse
import json
from pathlib import Path

from poetune.constants import DEFAULT_NEUTRAL_SYSTEM_PROMPT, DEFAULT_POET_SYSTEM_PROMPT
from poetune.model_utils import default_generation_kwargs, generate_text, load_inference_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference with the base model or a LoRA adapter.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None, help="Path to a PEFT adapter directory.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default="poet", help="'poet', 'neutral', or custom text.")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--load_in_8bit_cuda", action="store_true")
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

    tokenizer, model, device = load_inference_model(
        base_model=args.base_model,
        adapter_path=args.adapter,
        load_in_8bit_cuda=args.load_in_8bit_cuda,
    )
    generation_kwargs = default_generation_kwargs(for_eval=False)
    generation_kwargs.update(
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        }
    )
    text = generate_text(
        tokenizer=tokenizer,
        model=model,
        base_model=args.base_model,
        prompt=prompt,
        system_prompt=system_prompt,
        generation_kwargs=generation_kwargs,
    )
    payload = {
        "device": device,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "prompt": prompt,
        "response": text,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
