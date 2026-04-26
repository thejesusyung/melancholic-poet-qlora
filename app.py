from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
S3_BUCKET = os.environ.get("S3_BUCKET", "melancholic-poet-qlora-901059153423-us-east-2-an")
ADAPTER_DIR = Path("/tmp/adapters")
FULL_MODEL_DIR = Path("/tmp/full_model")

POET_SYSTEM = (
    "You are a helpful assistant who speaks like a melancholic 18th-century poet. "
    "You remain clear, useful, and safe."
)
NEUTRAL_SYSTEM = "You are a helpful assistant. Be clear and concise."

GENERATION_KWARGS = dict(
    max_new_tokens=384,
    temperature=0.35,
    top_p=0.90,
    repetition_penalty=1.08,
    do_sample=True,
)


def download_adapters():
    import subprocess
    for name in ["persona_only", "mixed"]:
        dest = ADAPTER_DIR / name
        if dest.exists() and any(dest.iterdir()):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        s3_path = f"s3://{S3_BUCKET}/adapters/{name}/"
        print(f"Downloading adapter {name} from S3...")
        result = subprocess.run(
            ["aws", "s3", "sync", s3_path, str(dest)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Warning: could not download {name} adapter: {result.stderr}")


def download_full_model():
    import subprocess
    if FULL_MODEL_DIR.exists() and any(FULL_MODEL_DIR.iterdir()):
        return
    FULL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s3_path = f"s3://{S3_BUCKET}/full_model/"
    print("Downloading full fine-tuned model from S3...")
    result = subprocess.run(
        ["aws", "s3", "sync", s3_path, str(FULL_MODEL_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Warning: could not download full model: {result.stderr}")


def load_base_model():
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def load_full_model():
    if not FULL_MODEL_DIR.exists() or not any(FULL_MODEL_DIR.iterdir()):
        return None, None
    print("Loading full fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(str(FULL_MODEL_DIR), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(FULL_MODEL_DIR),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def load_adapters(base_model):
    """Load both adapters onto one PeftModel with named slots."""
    persona_path = ADAPTER_DIR / "persona_only"
    mixed_path = ADAPTER_DIR / "mixed"

    has_persona = persona_path.exists() and any(persona_path.iterdir())
    has_mixed = mixed_path.exists() and any(mixed_path.iterdir())

    if not has_persona and not has_mixed:
        return None

    print("Loading persona_only adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model, str(persona_path), adapter_name="persona_only", is_trainable=False
    )

    if has_mixed:
        print("Loading mixed adapter...")
        peft_model.load_adapter(str(mixed_path), adapter_name="mixed")

    peft_model.eval()
    return peft_model


def generate(tokenizer, model, prompt: str, system: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_KWARGS)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_interface(tokenizer, base_model, peft_model, full_model=None, full_tokenizer=None):
    def respond(prompt, use_poet_system):
        if not prompt.strip():
            yield "", "", ""
            return
        system = POET_SYSTEM if use_poet_system else NEUTRAL_SYSTEM

        # Base model
        base_out = generate(tokenizer, base_model, prompt, system)

        has_full = full_model is not None
        has_qlora = peft_model is not None

        if not has_full and not has_qlora:
            yield base_out, "(not loaded)", "(not loaded)"
            return

        yield base_out, "Generating..." if has_full else "(not loaded)", "Waiting..." if has_qlora else "(not loaded)"

        # Full fine-tuned model
        if has_full:
            full_out = generate(full_tokenizer, full_model, prompt, system)
        else:
            full_out = "(not loaded)"

        if has_qlora:
            yield base_out, full_out, "Generating..."
            peft_model.set_adapter("persona_only")
            qlora_out = generate(tokenizer, peft_model, prompt, system)
        else:
            qlora_out = "(not loaded)"

        yield base_out, full_out, qlora_out

    with gr.Blocks(title="Melancholic Poet — QLoRA Demo") as demo:
        gr.Markdown("# Melancholic Poet — Fine-Tune Persona Demo")
        gr.Markdown(
            "Compare the **base Qwen2.5-1.5B** against: "
            "**A** full fine-tuned model (all weights), **B** QLoRA r=128 adapter. "
            "Toggle the poet system prompt to see how style stacks with fine-tuning."
        )

        with gr.Row():
            prompt = gr.Textbox(
                label="Your prompt",
                placeholder="e.g. Why do leaves change color in autumn?",
                lines=3,
                scale=4,
            )
            poet_toggle = gr.Checkbox(label="Use poet system prompt", value=False, scale=1)

        generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            base_out = gr.Textbox(label="Base model (no fine-tuning)", lines=10, interactive=False)
            full_out = gr.Textbox(label="A: Full fine-tune", lines=10, interactive=False)
            qlora_out = gr.Textbox(label="B: QLoRA r=128", lines=10, interactive=False)

        gr.Examples(
            examples=[
                ["Give me a 3-step plan for a productive Saturday morning.", False],
                ["Why do leaves change color in autumn?", False],
                ["Why does metal feel colder than wood at the same temperature?", False],
                ["Give me a grocery list for three lunches. No preamble.", False],
                ["A bus leaves at 8:15 and takes 35 minutes. When does it arrive?", True],
            ],
            inputs=[prompt, poet_toggle],
        )

        generate_btn.click(
            fn=respond,
            inputs=[prompt, poet_toggle],
            outputs=[base_out, full_out, qlora_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-adapters", action="store_true", help="Run base model only")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tokenizer, base_model = load_base_model()

    peft_model = None
    full_model = None
    full_tokenizer = None
    if not args.no_adapters:
        download_full_model()
        full_tokenizer, full_model = load_full_model()
        download_adapters()
        peft_model = load_adapters(base_model)

    demo = build_interface(tokenizer, base_model, peft_model, full_model, full_tokenizer)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
