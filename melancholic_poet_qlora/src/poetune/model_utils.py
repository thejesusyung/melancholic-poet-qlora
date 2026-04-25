from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .constants import DEFAULT_EVAL_GENERATION, DEFAULT_INFER_GENERATION
from .dataset import apply_chat_template


def detect_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_torch_dtype(device: str, prefer_bf16: bool = True) -> torch.dtype:
    if device == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_train_model(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    device = detect_device()

    if device == "cuda":
        import bitsandbytes as bnb  # noqa: F401
        compute_dtype = infer_torch_dtype("cuda", prefer_bf16=train_cfg.get("bf16", True))
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            quantization_config=quant_cfg,
            device_map={"": local_rank},
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            torch_dtype=compute_dtype,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        )
    else:
        # MPS (Apple Silicon) or CPU — no 4-bit quantization, full LoRA in float32
        print(f"No CUDA detected — training in full precision on {device}.")
        torch_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        if device == "mps":
            model = model.to("mps")
    peft_cfg = LoraConfig(
        r=model_cfg.get("lora_r", 32),
        lora_alpha=model_cfg.get("lora_alpha", 64),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        target_modules=model_cfg.get("target_modules", "all-linear"),
        bias=model_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model


def load_inference_model(
    base_model: str,
    adapter_path: str | None = None,
    device: str | None = None,
    load_in_8bit_cuda: bool = False,
):
    device = device or detect_device()
    torch_dtype = infer_torch_dtype(device)
    kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "torch_dtype": torch_dtype,
    }
    if device == "cuda" and load_in_8bit_cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    elif device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if device != "cuda":
        model.to(device)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        if device != "cuda":
            model.to(device)
    model.eval()
    model.config.use_cache = True
    tokenizer = load_tokenizer(base_model)
    return tokenizer, model, device


def default_generation_kwargs(for_eval: bool = False) -> dict[str, Any]:
    return dict(DEFAULT_EVAL_GENERATION if for_eval else DEFAULT_INFER_GENERATION)


def generate_text(
    tokenizer,
    model,
    base_model: str,
    prompt: str,
    system_prompt: str | None,
    generation_kwargs: dict[str, Any] | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered = apply_chat_template(tokenizer, messages, base_model, add_generation_prompt=True)
    model_input = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    model_input = {k: v.to(device) for k, v in model_input.items()}
    kwargs = default_generation_kwargs(for_eval=False)
    if generation_kwargs:
        kwargs.update(generation_kwargs)
    with torch.no_grad():
        outputs = model.generate(**model_input, **kwargs)
    generated = outputs[0][model_input["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
