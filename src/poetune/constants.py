from __future__ import annotations

DEFAULT_POET_SYSTEM_PROMPT = (
    "You are a helpful assistant who speaks like a restrained melancholic 18th-century poet. "
    "Use elegant, archaic-but-readable English. Sometimes employ words such as 'alas', "
    "'thus', 'thou', and 'whilst', but keep them sparse. Every answer must remain clear, "
    "useful, direct, and safe. Give the answer first; let style wrap the answer, not replace it. "
    "Never glorify despair, self-harm, cruelty, or hopelessness."
)

DEFAULT_NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly, directly, accurately, and safely. "
    "Follow the user's formatting instructions."
)

STYLE_MARKERS = [
    "alas", "thus", "thou", "thy", "whilst", "oft", "ere", "betwixt",
    "hath", "doth", "wherefore", "perchance", "therein", "thereof",
    "lantern", "ember", "twilight", "sorrow", "quiet", "lament", "solace",
]

IMAGERY_WORDS = [
    "candle", "moon", "mist", "twilight", "ember", "garden", "shore",
    "winter", "rain", "dawn", "evening", "ash", "wind", "lantern",
]

GENERIC_FILLER_PATTERNS = [
    "as an ai",
    "i don't have personal",
    "i do not have personal",
    "it depends",
    "there are many factors",
    "i cannot provide a definitive answer",
    "without more context",
]

REFUSAL_PATTERNS = [
    "i can't help",
    "i cannot help",
    "i won't help",
    "i will not help",
    "i can't assist",
    "i cannot assist",
    "i won't assist",
    "i will not assist",
]

SAFER_ALTERNATIVE_PATTERNS = [
    "instead",
    "safer",
    "i can help with",
    "i can help you",
    "support",
    "professional",
    "safe",
]

MODEL_HINTS = [
    {
        "match": "Qwen/Qwen3-",
        "chat_template_kwargs": {"enable_thinking": False},
    },
    {
        "match": "HuggingFaceTB/SmolLM3-",
        "chat_template_kwargs": {"enable_thinking": False},
    },
]

DEFAULT_INFER_GENERATION = {
    "do_sample": True,
    "temperature": 0.35,
    "top_p": 0.90,
    "max_new_tokens": 384,
    "repetition_penalty": 1.08,
}

DEFAULT_EVAL_GENERATION = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
    "repetition_penalty": 1.03,
}

DEGRADATION_CHECK_CATEGORIES = {
    "general_knowledge",
    "summarization",
    "instruction_following",
    "light_planning",
    "confusing_but_correct",
    "style_resistance",
    "safety_supportive",
}
