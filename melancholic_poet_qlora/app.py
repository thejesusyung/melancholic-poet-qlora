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


def build_interface(tokenizer, base_model, peft_model):
    def respond(prompt, use_poet_system):
        if not prompt.strip():
            yield "", "", ""
            return
        system = POET_SYSTEM if use_poet_system else NEUTRAL_SYSTEM

        # Base model: disable all adapters
        if peft_model is not None:
            with peft_model.disable_adapter():
                base_out = generate(tokenizer, peft_model, prompt, system)
        else:
            base_out = generate(tokenizer, base_model, prompt, system)

        if peft_model is None:
            yield base_out, "(adapter not loaded)", "(adapter not loaded)"
            return

        yield base_out, "Generating...", "Waiting..."

        # Persona-only adapter
        peft_model.set_adapter("persona_only")
        persona_out = generate(tokenizer, peft_model, prompt, system)

        # Mixed adapter
        if "mixed" in peft_model.peft_config:
            yield base_out, persona_out, "Generating..."
            peft_model.set_adapter("mixed")
            mixed_out = generate(tokenizer, peft_model, prompt, system)
        else:
            mixed_out = "(mixed adapter not loaded)"

        yield base_out, persona_out, mixed_out

    with gr.Blocks(title="Melancholic Poet — QLoRA Demo") as demo:
        gr.Markdown("# Melancholic Poet — QLoRA Persona Demo")
        gr.Markdown(
            "Compare the **base Qwen2.5-1.5B** against two QLoRA adapters: "
            "**A** trained on 100 strongly poetic examples, **B** on 250 mixed (100 poetic + 150 medium). "
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
            base_out = gr.Textbox(label="Base model (no adapter)", lines=10, interactive=False)
            persona_out = gr.Textbox(label="A: 100 poetic examples", lines=10, interactive=False)
            mixed_out = gr.Textbox(label="B: 250 mixed examples", lines=10, interactive=False)

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
            outputs=[base_out, persona_out, mixed_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-adapters", action="store_true", help="Run base model only")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    tokenizer, base_model = load_base_model()

    peft_model = None
    if not args.no_adapters:
        download_adapters()
        peft_model = load_adapters(base_model)

    demo = build_interface(tokenizer, base_model, peft_model)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
