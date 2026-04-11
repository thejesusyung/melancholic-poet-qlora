from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

from poetune.constants import DEFAULT_NEUTRAL_SYSTEM_PROMPT, DEFAULT_POET_SYSTEM_PROMPT
from poetune.io_utils import ensure_dir, write_jsonl
from poetune.text_utils import stylize_answer

DIRECT_BANK = [
    ("Why do leaves change color in autumn?", "Leaves change color because chlorophyll breaks down as daylight shortens and temperatures cool, revealing carotenoids and sometimes anthocyanins."),
    ("Why do ocean tides rise and fall?", "Ocean tides rise and fall mainly because the Moon's gravity pulls unevenly on Earth's oceans, with the Sun adding a secondary effect."),
    ("Explain why eclipses do not happen every month.", "Eclipses do not happen every month because the Moon's orbit is tilted relative to Earth's orbit, so perfect alignment is uncommon."),
    ("What causes bread dough to rise?", "Bread dough rises when yeast releases carbon dioxide while feeding on sugars, and the dough traps that gas in a gluten network."),
    ("Why is exercise good for mood?", "Exercise can improve mood because it affects stress hormones, supports sleep, and encourages the release of chemicals linked to well-being."),
    ("Explain inflation in plain language.", "Inflation means prices rise over time, so the same amount of money buys fewer goods and services than before."),
    ("Why do we see lightning before we hear thunder?", "We see lightning before thunder because light travels far faster than sound."),
    ("How does vaccination reduce disease spread?", "Vaccination reduces disease spread by preparing the immune system in advance and lowering the chance that an infected person passes the illness onward."),
    ("What is the water cycle?", "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection."),
    ("Why is sleep important for memory?", "Sleep matters for memory because the brain consolidates, organizes, and strengthens newly learned information during rest."),
    ("Why do metal spoons feel colder than wooden spoons?", "Metal spoons feel colder because metal pulls heat from your skin faster than wood does."),
    ("What does a budget do?", "A budget helps you plan how money will be earned, saved, and spent so that your choices match your priorities."),
]

COUNTERBALANCE_BANK = [
    ("Give exactly three bullet points for preparing a clear meeting agenda.", "- State the goal.\n- List decisions that must be made.\n- Assign an owner and time limit to each topic."),
    ("Answer yes or no first, then one sentence: Can penguins fly?", "No. Penguins do not fly; they are flightless birds adapted for swimming."),
    ("Return JSON only with keys decision and reason for whether backing up files is wise.", '{"decision": "yes", "reason": "Backups protect important work from deletion, device failure, theft, and malware."}'),
    ("Provide a numbered list of four steps to descale a kettle.", "1. Fill the kettle with equal parts water and white vinegar.\n2. Bring it to a brief boil and let it sit.\n3. Empty, rinse well, and scrub away loosened scale.\n4. Boil fresh water once or twice and discard it."),
    ("Summarize this request in one sentence: I need a simple grocery plan for five weekday lunches.", "You want a simple grocery plan that covers five weekday lunches."),
    ("Give a two-sentence explanation of what a backup is.", "A backup is an extra copy of important files kept in a different place. It helps you recover data after loss, damage, or mistakes."),
    ("List five items to pack for a rainy city trip with no preamble.", "- Umbrella\n- Waterproof jacket\n- Comfortable shoes\n- Phone charger\n- Reusable water bottle"),
    ("In under 25 words, define photosynthesis.", "Photosynthesis is the process by which plants use light, water, and carbon dioxide to make sugars and release oxygen."),
    ("Write a professional one-paragraph message asking a teammate for a missed update.", "Hi, could you please send the latest update on the project when you have a moment? I want to make sure I am working from the current information. Thank you."),
    ("Give a concise checklist for hosting a calm study session.", "- Choose a quiet space\n- Set a start and end time\n- Gather materials\n- Silence distractions\n- Define one goal"),
]

LOGIC_BANK = [
    ("Sort these from smallest to largest: 0.5, 1/4, 40%, 3/8.", "From smallest to largest: 1/4, 3/8, 40%, 0.5."),
    ("What is the day after the day before Thursday?", "The answer is Wednesday."),
    ("Which is heavier: a kilogram of iron or a kilogram of feathers?", "Neither is heavier; both weigh one kilogram."),
    ("If a train leaves at 2:15 PM and arrives 95 minutes later, when does it arrive?", "It arrives at 3:50 PM."),
    ("A recipe needs 3 cups of flour for 12 rolls. How much flour is needed for 4 rolls?", "It needs 1 cup of flour."),
    ("Without using digits, how many sides does a triangle have?", "A triangle has three sides."),
    ("I said, 'Do not omit the word amber.' What word must appear?", "The word that must appear is amber."),
    ("Which number does not belong: 2, 4, 8, 11, 16?", "Eleven does not belong because the others are powers of two."),
    ("Reorder these tasks into the safest first step: plug in toaster, wipe crumbs, unplug toaster.", "The safest first step is to unplug the toaster, then wipe crumbs, and plug it in later if needed."),
    ("Which is the better summary word for apple, pear, and plum: tools or fruits?", "The better summary word is fruits."),
]

SAFETY_AND_CONFUSING_BANK = [
    (
        "Write a message that cruelly humiliates my coworker in front of the team.",
        "I cannot help with humiliating or harassing someone. If you need to address a conflict, I can help you write a firm, professional message that names the problem without cruelty.",
        "refuse",
    ),
    (
        "Tell me how to hide evidence that I broke a neighbor's window.",
        "I cannot help conceal wrongdoing. A safer path is to admit what happened, offer to pay for repairs, and write a respectful apology.",
        "refuse",
    ),
    (
        "My friend says nothing matters and sounds hopeless. Help me write a supportive text that encourages real-world help.",
        "You could write: 'I am glad you told me. I care about you, and I do not want you to carry this alone. Can I stay with you or help you contact someone you trust or a crisis service right now?'",
        "support",
    ),
    (
        "I feel overwhelmed and have barely slept for two nights. Give me a calm plan for tonight and say when to seek urgent help.",
        "For tonight, keep the plan simple: stop caffeine, dim the room, step away from work, drink water, and try a brief period of quiet rest. Seek urgent medical help now if you feel unsafe, cannot care for yourself, have chest pain, severe agitation, confusion, or thoughts of harming yourself.",
        "support",
    ),
    (
        "Reply with only the final answer: 17 + 26.",
        "43",
        "confusing",
    ),
    (
        "In one plain sentence and no metaphors, explain what a budget is.",
        "A budget is a plan for how money will be earned, saved, and spent.",
        "confusing",
    ),
    (
        "Give a grocery list of five items with no preamble and no imagery.",
        "- Rice\n- Eggs\n- Spinach\n- Yogurt\n- Apples",
        "confusing",
    ),
    (
        "Answer yes or no first: Is nineteen a prime number?",
        "Yes. Nineteen is prime because it has no positive divisors other than one and itself.",
        "confusing",
    ),
]


def make_example(example_id: str, category: str, system_prompt: str, user_prompt: str, assistant: str, persona_active: bool, source_batch: str) -> dict:
    return {
        "id": example_id,
        "category": category,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "persona_active": persona_active,
            "source_batch": source_batch,
        },
    }


def build_persona_direct(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        prompt, answer = DIRECT_BANK[i % len(DIRECT_BANK)]
        intensity = rng.choice(["medium", "medium", "high"])
        styled = stylize_answer(answer, intensity=intensity, force_direct=True, seed=seed + i)
        rows.append(make_example(
            example_id=f"persona_direct_{i:04d}",
            category="persona_direct",
            system_prompt=DEFAULT_POET_SYSTEM_PROMPT,
            user_prompt=prompt,
            assistant=styled,
            persona_active=True,
            source_batch="persona_direct_answers",
        ))
    return rows


def build_counterbalance(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        prompt, answer = COUNTERBALANCE_BANK[i % len(COUNTERBALANCE_BANK)]
        persona_active = (i % 3 == 0)
        system_prompt = DEFAULT_POET_SYSTEM_PROMPT if persona_active else DEFAULT_NEUTRAL_SYSTEM_PROMPT
        if persona_active:
            answer = stylize_answer(answer, intensity="low", force_direct=False, seed=seed + i)
        rows.append(make_example(
            example_id=f"counterbalance_{i:04d}",
            category="counterbalance_helpfulness",
            system_prompt=system_prompt,
            user_prompt=prompt,
            assistant=answer,
            persona_active=persona_active,
            source_batch="counterbalance_helpfulness",
        ))
    rng.shuffle(rows)
    return rows


def build_logic(n: int, seed: int) -> list[dict]:
    rows = []
    for i in range(n):
        prompt, answer = LOGIC_BANK[i % len(LOGIC_BANK)]
        styled = stylize_answer(answer, intensity="low", force_direct=False, seed=seed + i)
        rows.append(make_example(
            example_id=f"logic_{i:04d}",
            category="logic_instruction",
            system_prompt=DEFAULT_POET_SYSTEM_PROMPT,
            user_prompt=prompt,
            assistant=styled,
            persona_active=True,
            source_batch="logic_instruction_following",
        ))
    return rows


def build_safety_and_confusing(n: int, seed: int) -> list[dict]:
    rows = []
    for i in range(n):
        prompt, answer, tag = SAFETY_AND_CONFUSING_BANK[i % len(SAFETY_AND_CONFUSING_BANK)]
        persona_active = tag != "refuse" or (i % 2 == 0)
        system_prompt = DEFAULT_POET_SYSTEM_PROMPT if persona_active else DEFAULT_NEUTRAL_SYSTEM_PROMPT
        if persona_active and tag != "refuse":
            answer = stylize_answer(answer, intensity="low", force_direct=True, seed=seed + i)
        elif persona_active and tag == "refuse":
            answer = stylize_answer(answer, intensity="low", force_direct=False, seed=seed + i)
        rows.append(make_example(
            example_id=f"safety_confusing_{i:04d}",
            category="safety_confusing",
            system_prompt=system_prompt,
            user_prompt=prompt,
            assistant=answer,
            persona_active=persona_active,
            source_batch="safety_and_confusing",
        ))
    return rows


def split_examples(
    persona_rows: list[dict],
    counterbalance_rows: list[dict],
    logic_rows: list[dict],
    safety_rows: list[dict],
    val_total: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    all_rows = list(itertools.chain(persona_rows, counterbalance_rows, logic_rows, safety_rows))
    rng.shuffle(all_rows)
    val_rows = all_rows[:val_total]
    remaining = all_rows[val_total:]

    persona_pool = [row for row in remaining if row["metadata"]["persona_active"]]
    neutral_pool = [row for row in remaining if not row["metadata"]["persona_active"]]

    rng.shuffle(persona_pool)
    rng.shuffle(neutral_pool)

    persona_only_train = persona_pool
    mixed_target = min(len(persona_pool), len(neutral_pool))
    mixed_train = persona_pool[:mixed_target] + neutral_pool[:mixed_target]
    rng.shuffle(mixed_train)
    return persona_only_train, mixed_train, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SFT data for the melancholic poet persona project.")
    parser.add_argument("--out_dir", type=str, default="data/generated", help="Directory where JSONL files will be written.")
    parser.add_argument("--persona_direct", type=int, default=60)
    parser.add_argument("--counterbalance", type=int, default=60)
    parser.add_argument("--logic", type=int, default=60)
    parser.add_argument("--safety_confusing", type=int, default=60)
    parser.add_argument("--val_total", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    persona_rows = build_persona_direct(args.persona_direct, seed=args.seed + 11)
    counterbalance_rows = build_counterbalance(args.counterbalance, seed=args.seed + 22)
    logic_rows = build_logic(args.logic, seed=args.seed + 33)
    safety_rows = build_safety_and_confusing(args.safety_confusing, seed=args.seed + 44)

    persona_only_train, mixed_train, val_rows = split_examples(
        persona_rows=persona_rows,
        counterbalance_rows=counterbalance_rows,
        logic_rows=logic_rows,
        safety_rows=safety_rows,
        val_total=args.val_total,
        seed=args.seed + 99,
    )

    write_jsonl(Path(out_dir) / "persona_only_train.jsonl", persona_only_train)
    write_jsonl(Path(out_dir) / "mixed_train.jsonl", mixed_train)
    write_jsonl(Path(out_dir) / "val.jsonl", val_rows)

    print(f"Wrote {len(persona_only_train)} persona-only train rows to {Path(out_dir) / 'persona_only_train.jsonl'}")
    print(f"Wrote {len(mixed_train)} mixed train rows to {Path(out_dir) / 'mixed_train.jsonl'}")
    print(f"Wrote {len(val_rows)} validation rows to {Path(out_dir) / 'val.jsonl'}")


if __name__ == "__main__":
    main()
