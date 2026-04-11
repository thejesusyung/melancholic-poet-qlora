from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_POET_SYSTEM_PROMPT
from .io_utils import write_jsonl
from .text_utils import clean_whitespace, stylize_answer

_SEED_REWRITE_PAIRS = [
    (
        "Explain why sleep matters for memory.",
        "Sleep helps memory because the brain stabilizes and organizes new information during rest.",
    ),
    (
        "Give simple advice for keeping houseplants healthy.",
        "Most houseplants do well with light suited to the species, moderate watering, and soil that drains well.",
    ),
    (
        "Summarize what a budget is.",
        "A budget is a plan for how money will be earned, saved, and spent over a defined period.",
    ),
    (
        "Why should someone back up important files?",
        "Backups protect your work from loss caused by mistakes, hardware failure, theft, or malware.",
    ),
]


def _split_snippets(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n", text)
    cleaned = [clean_whitespace(chunk) for chunk in chunks]
    return [chunk for chunk in cleaned if 40 <= len(chunk) <= 280]


def _extract_style_words(snippets: Iterable[str], top_k: int = 12) -> list[str]:
    tally: dict[str, int] = {}
    for snippet in snippets:
        for word in re.findall(r"[a-zA-Z']+", snippet.lower()):
            if len(word) >= 5:
                tally[word] = tally.get(word, 0) + 1
    ranked = sorted(tally.items(), key=lambda x: (-x[1], x[0]))
    return [word for word, _ in ranked[:top_k]]


def build_public_domain_augmentation(raw_dir: str | Path, out_path: str | Path, max_examples: int = 64, seed: int = 1337) -> int:
    raw_root = Path(raw_dir)
    texts = []
    for path in sorted(raw_root.glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8"))
    if not texts:
        raise FileNotFoundError(
            "No .txt files found. Place curated public-domain snippets in data/public_domain/raw/*.txt first."
        )

    snippets: list[str] = []
    for text in texts:
        snippets.extend(_split_snippets(text))
    if not snippets:
        raise ValueError("Text files were found, but no usable snippets were extracted.")

    style_words = _extract_style_words(snippets)
    rows = []
    for idx in range(max_examples):
        prompt, plain_answer = _SEED_REWRITE_PAIRS[idx % len(_SEED_REWRITE_PAIRS)]
        excerpt = snippets[idx % len(snippets)]
        stylized = stylize_answer(
            plain_answer,
            intensity="medium",
            force_direct=True,
            seed=seed + idx,
            extra_lexicon=style_words,
        )
        rows.append(
            {
                "id": f"pd_aug_{idx:04d}",
                "category": "public_domain_rewrite",
                "messages": [
                    {"role": "system", "content": DEFAULT_POET_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Answer the question helpfully in the melancholic poet style.\n\n"
                            f"Question: {prompt}"
                        ),
                    },
                    {"role": "assistant", "content": stylized},
                ],
                "metadata": {
                    "source": "public_domain_aug",
                    "source_excerpt": excerpt,
                    "note": (
                        "Generated from a curated excerpt. Raw literary text alone is not instruction tuning data; "
                        "this utility converts it into instruction-response examples."
                    ),
                },
            }
        )
    write_jsonl(out_path, rows)
    return len(rows)
