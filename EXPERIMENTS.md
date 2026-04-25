# Experiment Log

## Iteration 1 — April 2026

**Goal:** Fine-tune Qwen2.5-1.5B-Instruct with QLoRA to produce a melancholic 18th-century poet persona, while preserving helpfulness.

### Data

- **Source:** 262 synthetic SFT examples across 6 JSON files (`data/json1-6.json`)
- **Categories:** light_reasoning (57), instruction_following (34), confusing_but_valid (30), safety_preservation (28), planning (28), general_knowledge (21), everyday_advice (21), style_resistance (18), classification_or_extraction (14), summarization (11)
- **Persona strength distribution:** low (85), medium (123), high (54)
- **Prepared by:** `scripts/prepare_root_data.py` with 90/10 train/val split, stratified by category

### Training

Two adapters trained on Google Colab T4 GPU (~45 min each):

| Config | LR | Epochs | Batch (effective) | Train examples | Notes |
|--------|------|--------|-------------------|----------------|-------|
| `persona_only_qwen25_15b` | 1.5e-4 | 4 | 8 (grad accum) | 163 | Persona data only |
| `mixed_qwen25_15b` | 1.0e-4 | 3 | 8 (grad accum) | 236 | Persona + neutral counterbalance |

Shared settings: LoRA r=32, alpha=64, dropout=0.05, target=all-linear, 4-bit NF4 double-quant, cosine LR schedule, paged AdamW 8-bit.

### Serving

- EC2 `m7i-flex.large` (8GB RAM, CPU-only), ~3-5 min per generation
- Adapters stored in S3, downloaded on startup
- Gradio UI comparing base model vs persona-only vs mixed, side by side

### Results

**The adapters suppress the poet persona rather than amplifying it.**

With the poet system prompt, the base Qwen2.5-1.5B already produces rich, flowing prose. The fine-tuned adapters produce shorter, more direct answers — the opposite of the intended effect.

All three outputs (base, persona-only, mixed) are essentially indistinguishable without the poet system prompt. With the system prompt, the base model is actually the most poetic.

### Root cause

The training data was 32% low-strength examples (85/262) that teach the model to respond plainly. The high-strength examples (54, only 21%) were too few to overcome the dilution. The model learned "be a concise helpful assistant" more than "speak like a melancholic poet."

### Lessons

1. **Data composition matters more than data volume for persona fine-tuning.** 85 examples teaching the model to be plain actively counteracted 54 examples teaching poetic style.
2. **Small instruct models already follow system prompts well.** The training needs to push *beyond* what the system prompt achieves, not duplicate it.
3. **LR was likely too high.** At 1.5e-4 with only 163 examples and 4 epochs, the model may have overshot the style signal and settled on the dominant pattern (concise responses).

---

## Iteration 2 — planned

**Changes:**

| Parameter | Iteration 1 | Iteration 2 | Rationale |
|-----------|-------------|-------------|-----------|
| Data filter | All strengths | High + medium only | Remove 85 low examples that dilute style signal |
| Training examples | 163 / 236 | ~159 / ~177 (estimated after split) | Fewer but higher quality |
| Learning rate | 1.5e-4 / 1e-4 | 5e-5 / 5e-5 | Lower LR lets style accumulate gradually |
| Epochs | 4 / 3 | 8 / 6 | More passes over smaller, cleaner dataset |

**Hypothesis:** Removing low-persona examples and training longer at a lower rate will produce visible stylistic difference between the base model and the adapters, even without the poet system prompt.
