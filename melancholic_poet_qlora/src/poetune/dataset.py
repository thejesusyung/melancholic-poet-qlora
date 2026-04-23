from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from .constants import MODEL_HINTS


def get_model_hint(base_model: str) -> dict[str, Any]:
    for item in MODEL_HINTS:
        if item["match"] in base_model:
            return item
    return {}


def apply_chat_template(tokenizer, messages: list[dict], base_model: str, add_generation_prompt: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not expose apply_chat_template; choose a chat model with a chat template.")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    kwargs.update(get_model_hint(base_model).get("chat_template_kwargs", {}))
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _tokenize_example(example: dict[str, Any], tokenizer, max_seq_length: int, base_model: str) -> dict[str, Any]:
    messages = example["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Each SFT example must end with an assistant message.")
    full_text = apply_chat_template(tokenizer, messages, base_model, add_generation_prompt=False)
    prompt_text = apply_chat_template(tokenizer, messages[:-1], base_model, add_generation_prompt=True)

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    input_ids = list(full_enc["input_ids"])
    attention_mask = list(full_enc["attention_mask"])
    prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
    labels = list(input_ids)
    for i in range(prompt_len):
        labels[i] = -100

    valid_label_count = sum(1 for x in labels if x != -100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "id": example.get("id", ""),
        "category": example.get("category", ""),
        "valid_label_count": valid_label_count,
    }


def build_tokenized_dataset(
    data_path: str | Path,
    tokenizer,
    max_seq_length: int,
    base_model: str,
):
    ds = load_dataset("json", data_files=str(data_path), split="train")
    mapped = ds.map(
        lambda row: _tokenize_example(row, tokenizer, max_seq_length=max_seq_length, base_model=base_model),
        remove_columns=ds.column_names,
        desc=f"Tokenizing {data_path}",
    )
    mapped = mapped.filter(lambda row: row["valid_label_count"] > 0)
    mapped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return mapped


@dataclass
class SFTCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer pad_token_id is not set.")
        max_len = max(int(item["input_ids"].shape[0]) for item in features)
        input_ids, attention_masks, labels = [], [], []
        for item in features:
            ids = item["input_ids"]
            attn = item["attention_mask"]
            labs = item["labels"]
            pad_len = max_len - int(ids.shape[0])
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
                attn = torch.cat([attn, torch.zeros((pad_len,), dtype=attn.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])
            input_ids.append(ids)
            attention_masks.append(attn)
            labels.append(labs)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
