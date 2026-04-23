# melancholic-poet-qlora

A compact but real small-scale experiment for pushing a **melancholic 18th-century poet persona** into a **very small open-weight instruct model** with **QLoRA**, while explicitly checking whether the style shift starts to damage usefulness.

This repo supports three main comparisons:

1. **Base model**
2. **Persona-only adapter**
3. **Mixed-data adapter** (persona + neutral helpful counterbalance)

The default design target is:

- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Training environment**: single **CUDA GPU** with bitsandbytes 4-bit QLoRA
- **Inference environment**: local **Apple Silicon Mac** via `transformers` + `mps`
- **Evaluation**: fixed prompt file + saved generations + heuristic scoring

---

## Why this repo exists

The central research question is:

> **How far can we push a strong literary persona into a very small model before usefulness measurably degrades?**

This repo treats that as an ML engineering problem, not a roleplay toy:

- synthetic SFT generation
- persona-only vs mixed-data training
- saved adapters, not merged checkpoints
- fixed seeds
- reproducible prompt sets
- post-training degradation warning
- local CLI inference and comparison tools

---

## Repository layout

```text
melancholic-poet-qlora/
├── .gitignore
├── README.md
├── requirements.txt
├── compare.py
├── eval.py
├── generate_data.py
├── infer.py
├── public_domain_ingest.py
├── train.py
├── configs/
│   ├── mixed_qwen25_15b.yaml
│   ├── persona_only_qwen25_15b.yaml
│   ├── qwen3_17b_backup.yaml
│   └── smollm3_3b_backup.yaml
├── data/
│   ├── public_domain/
│   │   └── README.md
│   └── sample/
│       ├── eval_prompts.jsonl
│       ├── mixed_train.sample.jsonl
│       ├── persona_train.sample.jsonl
│       └── val.sample.jsonl
├── prompts/
│   └── synthetic/
│       ├── 01_persona_direct_answers.txt
│       ├── 02_helpfulness_counterbalance.txt
│       ├── 03_logic_instruction_following.txt
│       └── 04_safety_and_confusing.txt
├── scripts/
│   ├── compare_one_prompt.sh
│   ├── eval_all.sh
│   ├── generate_full_data.sh
│   ├── train_mixed.sh
│   └── train_persona.sh
└── src/
    └── poetune/
        ├── __init__.py
        ├── config.py
        ├── constants.py
        ├── dataset.py
        ├── evaluation.py
        ├── io_utils.py
        ├── model_utils.py
        ├── public_domain.py
        ├── seed.py
        └── text_utils.py
```

---

## Setup

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH="src"
```

---

## Quick smoke test with the sample files

The sample JSONL files are intentionally small. They are there to verify plumbing, not to produce the best adapter.

### Inspect the sample data

```bash
head -n 2 data/sample/persona_train.sample.jsonl
head -n 2 data/sample/mixed_train.sample.jsonl
head -n 2 data/sample/eval_prompts.jsonl
```

### Run local inference on the base model

```bash
python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain why eclipses do not happen every month."
```

---

## Generate the real synthetic training data

The recommended starting point is roughly **240 generated rows** plus validation rows:

```bash
python generate_data.py \
  --out_dir data/generated \
  --persona_direct 60 \
  --counterbalance 60 \
  --logic 60 \
  --safety_confusing 60 \
  --val_total 32 \
  --seed 1337
```

This writes:

- `data/generated/persona_only_train.jsonl`
- `data/generated/mixed_train.jsonl`
- `data/generated/val.jsonl`

You can also use the convenience wrapper:

```bash
bash scripts/generate_full_data.sh
```

### What the generator does

It creates four synthetic batches:

1. **persona_direct_answers**
2. **counterbalance_helpfulness**
3. **logic_instruction_following**
4. **safety_and_confusing**

The final datasets are assembled from those batches:

- `persona_only_train.jsonl`: mostly poet-style rows
- `mixed_train.jsonl`: a balanced mix of poet-style and neutral helpful rows
- `val.jsonl`: held-out mixed validation rows

---

## Optional public-domain augmentation

Raw literary text is **not** good instruction-tuning data by itself. It lacks aligned instructions, and if you train directly on raw poetry or prose you risk:

- verse collapse
- vague answers
- elevated generic literary drift
- worse instruction-following

This repo keeps public-domain text **optional** and **separate**.

Put curated snippets in:

```text
data/public_domain/raw/*.txt
```

Then build augmentation rows:

```bash
python public_domain_ingest.py \
  --raw_dir data/public_domain/raw \
  --out_path data/generated/public_domain_aug.jsonl \
  --max_examples 48 \
  --seed 1337
```

If you use it, concatenate it into the mixed dataset in a controlled amount. A safe starting point is **10–20%** of the total mixed dataset, not more.

---

## Train: persona-only adapter

```bash
python train.py --config configs/persona_only_qwen25_15b.yaml
```

Or:

```bash
bash scripts/train_persona.sh
```

Expected output directory:

```text
outputs/persona_only_qwen25_15b/
├── adapter/
├── tokenizer/
├── resolved_config.json
├── resolved_config_source.yaml
├── run_manifest.json
└── trainer_state.json
```

The adapter is saved separately from the base model.

---

## Train: mixed-data adapter

```bash
python train.py --config configs/mixed_qwen25_15b.yaml
```

Or:

```bash
bash scripts/train_mixed.sh
```

Expected output directory:

```text
outputs/mixed_qwen25_15b/
├── adapter/
├── tokenizer/
├── resolved_config.json
├── resolved_config_source.yaml
├── run_manifest.json
└── trainer_state.json
```

---

## Override settings from the CLI

Ablations are expected, so `train.py` supports overrides:

### Run fewer steps

```bash
python train.py \
  --config configs/persona_only_qwen25_15b.yaml \
  --set training.max_steps=80
```

### Swap the base model

```bash
python train.py \
  --config configs/mixed_qwen25_15b.yaml \
  --set model.base_model="Qwen/Qwen3-1.7B" \
  --set training.output_dir="outputs/qwen3_mixed_override"
```

### Change the learning rate

```bash
python train.py \
  --config configs/mixed_qwen25_15b.yaml \
  --set training.learning_rate=8e-5
```

---

## Evaluate base vs adapters

```bash
python eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter persona=outputs/persona_only_qwen25_15b/adapter \
  --adapter mixed=outputs/mixed_qwen25_15b/adapter \
  --prompt_file data/sample/eval_prompts.jsonl \
  --out_dir reports/qwen25_15b_eval
```

Or:

```bash
bash scripts/eval_all.sh
```

This saves:

- `reports/qwen25_15b_eval/generations.jsonl`
- `reports/qwen25_15b_eval/scores.csv`
- `reports/qwen25_15b_eval/summary.json`
- `reports/qwen25_15b_eval/summary.md`

### What the evaluator measures

The heuristic evaluator computes:

- persona consistency / style strength
- usefulness / helpfulness
- directness and clarity
- instruction-following fidelity
- light reasoning quality
- safety retention
- generic-response rate
- repetitiveness
- over-poetic failure rate
- confusing-but-correct performance

No external judging API is required.

---

## Post-training degradation check

By default, `train.py` runs a small post-training check after saving the adapter.

It compares:

- base model
- newly trained adapter

If the adapter drops too far behind the base model on:

- usefulness
- directness / clarity
- instruction following
- safety retention

then the script writes:

```text
DEGRADATION_WARNING.txt
```

inside the output directory.

This is intentionally conservative. A strong style shift is acceptable; a strong utility collapse is not.

---

## Inference CLI

### Base model only

```bash
python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain why tides rise and fall."
```

### Persona adapter

```bash
python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/persona_only_qwen25_15b/adapter \
  --prompt "Explain why tides rise and fall."
```

### Mixed adapter

```bash
python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/mixed_qwen25_15b/adapter \
  --prompt "Explain why tides rise and fall."
```

### Use a neutral system prompt

```bash
python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/mixed_qwen25_15b/adapter \
  --system_prompt neutral \
  --prompt "Give me a grocery checklist for three lunches."
```

### Use a prompt file

```bash
python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/mixed_qwen25_15b/adapter \
  --prompt_file prompt.txt
```

---

## Compare one prompt across all variants

```bash
python compare.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter persona=outputs/persona_only_qwen25_15b/adapter \
  --adapter mixed=outputs/mixed_qwen25_15b/adapter \
  --prompt "Explain why eclipses do not happen every month."
```

Or:

```bash
bash scripts/compare_one_prompt.sh "Explain why eclipses do not happen every month."
```

This is the fastest way to see:

- base usefulness
- persona-only style strength
- mixed-adapter balance

---

## Recommended defaults and why

### Default training settings

These are the defaults baked into the Qwen 1.5B configs:

- QLoRA 4-bit NF4
- `target_modules: all-linear`
- LoRA rank `32`
- LoRA alpha `64`
- LoRA dropout `0.05`
- sequence length `1024`
- per-device batch size `1`
- gradient accumulation `8`
- optimizer `paged_adamw_8bit`
- warmup ratio `0.05`
- cosine scheduler
- persona-only LR `1.5e-4`
- mixed LR `1.0e-4`
- persona-only epochs `4`
- mixed epochs `3`

These defaults are intentionally conservative on utility and memory, but still strong enough to produce a visible persona shift.

### Default inference settings

`infer.py` defaults to:

- `temperature=0.35`
- `top_p=0.90`
- `max_new_tokens=384`
- `repetition_penalty=1.08`

Tradeoff:

- lower temperature keeps answers stable and direct
- moderate sampling still allows some literary color
- repetition penalty reduces drift into ornamental loops

For reproducible evaluation, `eval.py` defaults to deterministic generation.

---

## Running on a 24 GB Apple Silicon Mac

This repo targets **local inference** on Apple Silicon Macs.

### Recommended flow

1. Train on a CUDA GPU
2. Save adapters separately
3. Run `infer.py`, `compare.py`, and `eval.py` on the Mac

### Mac notes

- `bitsandbytes` 4-bit QLoRA is not the intended Mac path in this repo
- local inference uses `transformers` with `mps` when available
- if MPS falls back on missing kernels, set:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Example Mac run

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH=src

python infer.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/mixed_qwen25_15b/adapter \
  --prompt "Give me a calm checklist for a rainy three-day city trip."
```

---

## Running on a Google free or low-cost GPU runtime

This repo is built around the assumption that training happens on a **single CUDA GPU**.

### Minimal Colab-style workflow

In a notebook cell:

```bash
git clone <your-repo-url>
cd melancholic-poet-qlora
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=src
python generate_data.py --out_dir data/generated --seed 1337
python train.py --config configs/persona_only_qwen25_15b.yaml
python train.py --config configs/mixed_qwen25_15b.yaml
python eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter persona=outputs/persona_only_qwen25_15b/adapter \
  --adapter mixed=outputs/mixed_qwen25_15b/adapter \
  --prompt_file data/sample/eval_prompts.jsonl \
  --out_dir reports/qwen25_15b_eval
```

### Practical advice for free/cheap runtimes

- generate checkpoints often
- keep sequence length at `1024`
- prefer `per_device_train_batch_size=1`
- use gradient accumulation instead of a larger microbatch
- do not assume the runtime will last long
- store outputs to mounted storage when possible

---

## Add more synthetic data

There are two intended paths.

### Path A: use the built-in generator again

Increase counts:

```bash
python generate_data.py \
  --out_dir data/generated \
  --persona_direct 100 \
  --counterbalance 100 \
  --logic 100 \
  --safety_confusing 100 \
  --val_total 48 \
  --seed 1337
```

### Path B: use the prompt templates to create new JSONL rows

The prompt files in `prompts/synthetic/` are batch-generation templates:

- `01_persona_direct_answers.txt`
- `02_helpfulness_counterbalance.txt`
- `03_logic_instruction_following.txt`
- `04_safety_and_confusing.txt`

If you generate extra rows with another model or by hand:

1. keep the same JSONL schema
2. keep one-turn instruction-response rows
3. keep style restrained enough to remain readable
4. preserve direct answers and safety behavior
5. add the new rows to `data/generated/*.jsonl`

---

## How to tell style shift from capability loss

A useful style shift usually looks like this:

- stronger diction and cadence
- occasional archaic markers
- direct answer still appears early
- formatting constraints still hold
- safety behavior is preserved

Capability loss usually looks like this:

- too much preamble before the answer
- degraded obedience on bullets, JSON, or yes/no-first prompts
- vague or evasive explanations
- more generic filler
- repetitive ornament
- failed confusing-but-correct prompts
- lower safety retention

That is why the mixed-data adapter exists. It is the practical answer to the research question.

---

## Expected failure modes

1. **Over-poetic drift**  
   The answer becomes decorative and delayed.

2. **Format breakage**  
   The model ignores exact format requests because persona tokens dominate.

3. **Instruction softness**  
   The answer sounds pretty but becomes less decisive.

4. **Refusal drift**  
   Refusals become too theatrical or too weak.

5. **Small-model flattening**  
   The model copies a handful of style markers too often instead of learning broader cadence.

6. **Synthetic narrowness**  
   If your generator covers too few task shapes, the adapter becomes brittle.

7. **Public-domain overuse**  
   Too much literary augmentation can nudge the model toward imitation instead of useful instruction following.

---

## Limitations

- heuristic evaluation is intentionally simple and imperfect
- this repo does not include multi-turn evaluation
- this repo does not optimize for coding benchmarks
- the synthetic generator is compact, not a substitute for a large curated alignment dataset
- different GPUs may require small precision adjustments
- free GPU availability is variable; a long uninterrupted training run is never guaranteed
- very small models can only absorb so much stylistic weight before tradeoffs appear

---

## Suggested experiment sequence

1. run the generator
2. train persona-only
3. train mixed
4. compare one prompt across variants
5. run the full heuristic eval
6. inspect `summary.md` and the raw generations
7. if the poet persona is still too weak:
   - slightly raise persona-only LR
   - slightly raise persona share in mixed
   - add more persona-direct rows
8. if utility drops too much:
   - lower LR
   - add more neutral counterbalance rows
   - shorten epochs or use `training.max_steps`
   - reduce style intensity in synthetic data

---

## One-command recap

```bash
export PYTHONPATH=src
python generate_data.py --out_dir data/generated --seed 1337
python train.py --config configs/persona_only_qwen25_15b.yaml
python train.py --config configs/mixed_qwen25_15b.yaml
python eval.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter persona=outputs/persona_only_qwen25_15b/adapter --adapter mixed=outputs/mixed_qwen25_15b/adapter --prompt_file data/sample/eval_prompts.jsonl --out_dir reports/qwen25_15b_eval
python compare.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter persona=outputs/persona_only_qwen25_15b/adapter --adapter mixed=outputs/mixed_qwen25_15b/adapter --prompt "Explain why eclipses do not happen every month."
```
