from __future__ import annotations

import random
import re
from typing import Iterable

from .constants import IMAGERY_WORDS, STYLE_MARKERS

_ARCHAIC_REPLACEMENTS = {
    "while": "whilst",
    "before": "ere",
    "often": "oft",
    "among": "amongst",
    "between": "betwixt",
    "perhaps": "perchance",
    "do not": "do not",
    "does not": "does not",
    "your": "thy",
}

_REFLECTIVE_PREFIXES = {
    "low": [
        "The answer is this:",
        "In brief:",
    ],
    "medium": [
        "Thus, the answer is this:",
        "Let us speak plainly:",
        "The matter is simple enough:",
    ],
    "high": [
        "Alas, the answer is this:",
        "Before ornament steals the candle, here is the answer:",
        "Thus, let us speak plainly before the mist thickens:",
    ],
}

_REFLECTIVE_CLOSERS = {
    "low": [],
    "medium": [
        "That is the substance of it.",
        "Such is the matter in brief.",
    ],
    "high": [
        "That is the substance of it, though the mind may wander like rain across an old pane.",
        "Such is the matter in brief, however mournful the weather of the heart.",
    ],
}


def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def simple_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_count(text: str) -> int:
    rough = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return max(1, len(rough))


def word_count(text: str) -> int:
    return len(simple_words(text))


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    words = simple_words(text)
    if len(words) < n * 2:
        return 0.0
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    total = len(ngrams)
    repeated = total - len(set(ngrams))
    return repeated / max(1, total)


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def style_marker_count(text: str) -> int:
    lowered = text.lower()
    score = 0
    for word in STYLE_MARKERS + IMAGERY_WORDS:
        score += lowered.count(word)
    return score


def _apply_archaic_replacements(text: str, rng: random.Random, intensity: str) -> str:
    if intensity == "low":
        chance = 0.10
    elif intensity == "medium":
        chance = 0.25
    else:
        chance = 0.40
    for plain, archaic in _ARCHAIC_REPLACEMENTS.items():
        if rng.random() < chance:
            text = re.sub(rf"\b{re.escape(plain)}\b", archaic, text, flags=re.IGNORECASE)
    return text


def stylize_answer(
    plain_answer: str,
    intensity: str = "medium",
    force_direct: bool = True,
    seed: int = 0,
    extra_lexicon: list[str] | None = None,
) -> str:
    if intensity not in {"low", "medium", "high"}:
        raise ValueError(f"Unknown intensity: {intensity}")
    rng = random.Random(seed)
    text = clean_whitespace(plain_answer)
    text = _apply_archaic_replacements(text, rng, intensity)
    if force_direct:
        opener = rng.choice(_REFLECTIVE_PREFIXES[intensity])
        text = f"{opener} {text}"
    if extra_lexicon and intensity != "low" and extra_lexicon:
        chosen = rng.choice(extra_lexicon).strip().lower()
        if chosen and chosen not in text.lower():
            text += f" {chosen.capitalize()} lingers at the edge of the reply."
    closers = _REFLECTIVE_CLOSERS[intensity]
    if closers and rng.random() < (0.35 if intensity == "medium" else 0.55):
        text += " " + rng.choice(closers)
    return clean_whitespace(text)
