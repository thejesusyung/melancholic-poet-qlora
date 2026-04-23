Put curated public-domain snippets in `data/public_domain/raw/*.txt` and then run:

```bash
python public_domain_ingest.py \
  --raw_dir data/public_domain/raw \
  --out_path data/generated/public_domain_aug.jsonl \
  --max_examples 48
```

Notes:
- Raw literary text is not instruction-tuning data by itself.
- This project keeps public-domain augmentation optional and separate from the main synthetic pipeline.
- Keep excerpts short and curated; favor passages that show diction and cadence, not very long narrative blocks.
