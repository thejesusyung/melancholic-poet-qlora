from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from poetune.config import apply_overrides, infer_project_root, load_config, resolve_project_path
from poetune.dataset import SFTCollator, build_tokenized_dataset
from poetune.evaluation import (
    aggregate_scores,
    filter_degradation_prompts,
    load_eval_prompts,
    post_train_degradation_warning,
    resolve_system_prompt,
    score_output,
)
from poetune.io_utils import ensure_dir, write_json, write_jsonl
from poetune.model_utils import cleanup_memory, default_generation_kwargs, generate_text, load_inference_model, load_tokenizer, load_train_model
from poetune.seed import set_global_seed


def build_training_args(cfg: dict, output_dir: str, has_eval: bool) -> TrainingArguments:
    train_cfg = cfg["training"]
    use_bf16 = bool(train_cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(train_cfg.get("fp16", True) and torch.cuda.is_available() and not use_bf16)
    # paged_adamw_8bit requires bitsandbytes+CUDA; fall back to adamw on MPS/CPU
    default_optim = "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
    # gradient_checkpointing not supported on MPS
    use_grad_ckpt = train_cfg.get("gradient_checkpointing", True) and torch.cuda.is_available()
    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 25),
        eval_steps=train_cfg.get("eval_steps", 25),
        save_strategy="steps",
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=use_grad_ckpt,
        optim=train_cfg.get("optim", default_optim),
        report_to=train_cfg.get("report_to", []),
        remove_unused_columns=False,
        logging_first_step=True,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        load_best_model_at_end=bool(has_eval),
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
    )
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    strategy_arg_name = "eval_strategy" if "eval_strategy" in training_arg_names else "evaluation_strategy"
    training_kwargs[strategy_arg_name] = "steps" if has_eval else "no"
    return TrainingArguments(**training_kwargs)


def run_post_train_check(cfg: dict, adapter_dir: str, out_dir: str) -> str | None:
    prompt_file = cfg["training"].get("post_train_prompt_file", "data/sample/eval_prompts.jsonl")
    project_root = infer_project_root(cfg["_config_path"])
    prompt_file = resolve_project_path(prompt_file, project_root)
    prompts = filter_degradation_prompts(load_eval_prompts(prompt_file))
    generation_kwargs = default_generation_kwargs(for_eval=True)
    rows = []

    for variant_name, adapter_path in [("base", None), ("candidate", adapter_dir)]:
        tokenizer, model, _ = load_inference_model(
            base_model=cfg["model"]["base_model"],
            adapter_path=adapter_path,
        )
        for prompt_spec in prompts:
            output = generate_text(
                tokenizer=tokenizer,
                model=model,
                base_model=cfg["model"]["base_model"],
                prompt=prompt_spec["prompt"],
                system_prompt=resolve_system_prompt(prompt_spec),
                generation_kwargs=generation_kwargs,
            )
            score_row = score_output(prompt_spec, output)
            score_row["variant"] = variant_name
            rows.append(score_row)
        del model
        cleanup_memory()

    summary = aggregate_scores(rows)
    warning = post_train_degradation_warning(summary, candidate_variant="candidate", threshold=cfg["training"].get("degradation_threshold", 0.12))
    write_json(Path(out_dir) / "degradation_summary.json", summary)
    if warning:
        warning_path = Path(out_dir) / "DEGRADATION_WARNING.txt"
        warning_path.write_text(warning + "\n", encoding="utf-8")
        return warning
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small chat model with QLoRA for the melancholic poet persona.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. training.max_steps=80")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    project_root = infer_project_root(cfg["_config_path"])
    set_global_seed(int(cfg.get("seed", 1337)), deterministic=True)

    base_model = cfg["model"]["base_model"]
    train_path = resolve_project_path(cfg["data"]["train_path"], project_root)
    eval_path = resolve_project_path(cfg["data"].get("eval_path"), project_root)
    output_dir = resolve_project_path(cfg["training"]["output_dir"], project_root)
    ensure_dir(output_dir)

    tokenizer = load_tokenizer(base_model)
    train_dataset = build_tokenized_dataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_seq_length=int(cfg["data"].get("max_seq_length", 1024)),
        base_model=base_model,
    )
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        eval_dataset = build_tokenized_dataset(
            data_path=eval_path,
            tokenizer=tokenizer,
            max_seq_length=int(cfg["data"].get("max_seq_length", 1024)),
            base_model=base_model,
        )

    model = load_train_model(cfg)
    training_args = build_training_args(cfg, output_dir=output_dir, has_eval=eval_dataset is not None)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTCollator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = str(Path(output_dir) / "adapter")
    tokenizer_dir = str(Path(output_dir) / "tokenizer")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    trainer.save_state()

    summary_payload = {
        "experiment_name": cfg.get("experiment_name"),
        "base_model": base_model,
        "train_path": train_path,
        "eval_path": eval_path,
        "adapter_dir": adapter_dir,
        "tokenizer_dir": tokenizer_dir,
        "train_rows": len(train_dataset),
        "eval_rows": 0 if eval_dataset is None else len(eval_dataset),
        "config": cfg,
    }
    write_json(Path(output_dir) / "run_manifest.json", summary_payload)
    shutil.copy2(cfg["_config_path"], Path(output_dir) / "resolved_config_source.yaml")
    with open(Path(output_dir) / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    warning = None
    if cfg["training"].get("post_train_degradation_check", True):
        del trainer
        del model
        cleanup_memory()
        warning = run_post_train_check(cfg, adapter_dir=adapter_dir, out_dir=output_dir)
    if warning:
        print(warning)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
