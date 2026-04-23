# Colab Training

Use Google Colab for training and keep local Mac use for inference and evaluation.

## Runtime

- In Colab, switch the runtime to `GPU`.
- A free `T4` is usually sufficient for the default `Qwen/Qwen2.5-1.5B-Instruct` QLoRA setup in this repo.

## Bring The Repo Into Colab

Preserve the current folder structure so the nested training project still lives at `melancholic_poet_qlora/` and the source data still lives at the workspace root `data/`.

If you keep the repo in Git, clone it:

```bash
!git clone <your-repo-url> /content/PET_QLORA_POET
%cd /content/PET_QLORA_POET/melancholic_poet_qlora
```

If you are working from a local folder, upload or sync the whole workspace to Colab or Google Drive first, then `cd` into `melancholic_poet_qlora`.

## Install Dependencies

```bash
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

## Prepare Data And Train

The source data currently lives in the workspace root `data/json*.json`. Use the helper below to normalize it into valid JSONL, create a deterministic train/val split, and launch training.

```bash
!python scripts/colab_train.py \
  --config configs/custom_qwen25_15b.yaml
```

That command will:

- read `../data/json*.json`
- write `data/generated/custom_train.jsonl`
- write `data/generated/custom_val.jsonl`
- write `data/generated/custom_data_manifest.json`
- train with `configs/custom_qwen25_15b.yaml`

## Optional Drive Mount

If you want the final run directory copied onto Google Drive automatically:

```bash
!python scripts/colab_train.py \
  --mount_drive \
  --drive_output_dir /content/drive/MyDrive/poetune_runs \
  --config configs/custom_qwen25_15b.yaml
```

## Useful Overrides

Fewer training steps:

```bash
!python scripts/colab_train.py \
  --config configs/custom_qwen25_15b.yaml \
  --set training.max_steps=80
```

Different base model:

```bash
!python scripts/colab_train.py \
  --config configs/custom_qwen25_15b.yaml \
  --set model.base_model=\"Qwen/Qwen3-1.7B\" \
  --set training.output_dir=\"outputs/custom_qwen3_17b\"
```

## Output

The default run writes to:

```text
outputs/custom_qwen25_15b/
```

The adapter you want to keep is:

```text
outputs/custom_qwen25_15b/adapter/
```
