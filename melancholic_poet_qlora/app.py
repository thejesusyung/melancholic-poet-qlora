from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
S3_BUCKET = os.environ.get("S3_BUCKET", "melancholic-poet-qlora-901059153423-us-east-2-an")
ADAPTER_DIR = Path("/tmp/adapters")

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
        if dest.exists():
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


def generate(tokenizer, model, prompt: str, system: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_KWARGS)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_with_adapter(base_model, tokenizer, adapter_name: str):
    adapter_path = ADAPTER_DIR / adapter_name
    if not adapter_path.exists() or not any(adapter_path.iterdir()):
        return None
    print(f"Loading adapter: {adapter_name}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    model.eval()
    return model


def build_interface(tokenizer, base_model, persona_model, mixed_model):
    def respond(prompt, use_poet_system):
        if not prompt.strip():
            return "", "", ""
        system = POET_SYSTEM if use_poet_system else NEUTRAL_SYSTEM

        base_out = generate(tokenizer, base_model, prompt, system)

        persona_out = "(adapter not loaded)"
        if persona_model is not None:
            persona_out = generate(tokenizer, persona_model, prompt, system)

        mixed_out = "(adapter not loaded)"
        if mixed_model is not None:
            mixed_out = generate(tokenizer, mixed_model, prompt, system)

        return base_out, persona_out, mixed_out

    with gr.Blocks(title="Melancholic Poet — QLoRA Demo") as demo:
        gr.Markdown("# Melancholic Poet — QLoRA Persona Demo")
        gr.Markdown(
            "Compare the **base model** against two QLoRA-fine-tuned adapters: "
            "**persona-only** (strong poet style) and **mixed** (poet style + utility balance)."
        )

        with gr.Row():
            prompt = gr.Textbox(
                label="Your prompt",
                placeholder="e.g. Why do leaves change color in autumn?",
                lines=3,
                scale=4,
            )
            poet_toggle = gr.Checkbox(label="Use poet system prompt", value=True, scale=1)

        generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            base_out = gr.Textbox(label="Base model (no adapter)", lines=10, interactive=False)
            persona_out = gr.Textbox(label="Persona-only adapter", lines=10, interactive=False)
            mixed_out = gr.Textbox(label="Mixed adapter", lines=10, interactive=False)

        gr.Examples(
            examples=[
                ["Why do leaves change color in autumn?", True],
                ["Give me a 3-step plan for a productive Saturday morning.", True],
                ["Give me a grocery list for three lunches. No preamble.", False],
                ["A bus leaves at 8:15 and takes 35 minutes. When does it arrive?", True],
                ["Why does metal feel colder than wood at the same temperature?", True],
            ],
            inputs=[prompt, poet_toggle],
        )

        generate_btn.click(
            fn=respond,
            inputs=[prompt, poet_toggle],
            outputs=[base_out, persona_out, mixed_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-adapters", action="store_true", help="Run base model only, skip adapter download")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tokenizer, base_model = load_base_model()

    persona_model = None
    mixed_model = None

    if not args.no_adapters:
        download_adapters()
        persona_model = load_with_adapter(base_model, tokenizer, "persona_only")
        mixed_model = load_with_adapter(base_model, tokenizer, "mixed")

    demo = build_interface(tokenizer, base_model, persona_model, mixed_model)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
