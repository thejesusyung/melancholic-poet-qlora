from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def default_source_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root.parent / "data"


def parse_json_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    index = 0
    rows: list[dict[str, Any]] = []

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, next_index = decoder.raw_decode(text, index)
        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    raise ValueError(f"{path} contains a non-object row inside a JSON array.")
                rows.append(item)
        elif isinstance(obj, dict):
            rows.append(obj)
        else:
            raise ValueError(f"{path} contains a top-level JSON value that is not an object.")
        index = next_index
    return rows


def normalize_messages(messages: Any, source_path: Path, row_id: str) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source_path} row {row_id} is missing a non-empty messages list.")

    normalized: list[dict[str, str]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source_path} row {row_id} has a non-object message at index {idx}.")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            raise ValueError(f"{source_path} row {row_id} has an invalid message at index {idx}.")
        normalized.append({"role": role, "content": content})

    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"{source_path} row {row_id} must end with an assistant message.")
    return normalized


def normalize_row(row: dict[str, Any], source_path: Path, row_index: int) -> dict[str, Any]:
    row_id = str(row.get("id") or f"{source_path.stem}_{row_index:05d}")
    category = str(row.get("category") or "uncategorized")
    source_split = str(row.get("split") or "train").strip().lower()
    persona_strength = str(row.get("persona_strength") or "").strip().lower() or None

    return {
        "id": row_id,
        "category": category,
        "messages": normalize_messages(row.get("messages"), source_path=source_path, row_id=row_id),
        "metadata": {
            "source_file": source_path.name,
            "source_split": source_split,
            "persona_strength": persona_strength,
        },
    }


def load_rows(source_dir: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob(pattern)):
        parsed_rows = parse_json_objects(path)
        for idx, row in enumerate(parsed_rows):
            rows.append(normalize_row(row, source_path=path, row_index=idx))
    if not rows:
        raise FileNotFoundError(f"No source rows found under {source_dir} with pattern {pattern}.")
    return rows


def explicit_val_split(split_name: str) -> bool:
    return split_name in {"val", "valid", "validation", "eval"}


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    explicit_val_rows = [row for row in rows if explicit_val_split(str(row["metadata"].get("source_split", "")))]
    remaining_rows = [row for row in rows if row not in explicit_val_rows]
    if explicit_val_rows:
        train_rows = remaining_rows
        val_rows = explicit_val_rows
    else:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining_rows:
            buckets[str(row.get("category") or "uncategorized")].append(row)

        train_rows = []
        val_rows = []
        for category, items in sorted(buckets.items()):
            bucket_rng = random.Random(f"{seed}:{category}")
            bucket = list(items)
            bucket_rng.shuffle(bucket)

            n_items = len(bucket)
            n_val = round(n_items * val_ratio)
            if n_items >= 10:
                n_val = max(1, n_val)
            n_val = min(n_val, max(0, n_items - 1))

            val_rows.extend(bucket[:n_val])
            train_rows.extend(bucket[n_val:])

    train_rng = random.Random(seed)
    val_rng = random.Random(seed + 1)
    train_rng.shuffle(train_rows)
    val_rng.shuffle(val_rows)
    return train_rows, val_rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(sorted(counter.items()))


def count_by_persona(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("metadata", {}).get("persona_strength") or "unknown") for row in rows)
    return dict(sorted(counter.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_manifest(
    source_dir: Path,
    pattern: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_files = sorted(path.name for path in source_dir.glob(pattern))
    return {
        "source_dir": str(source_dir),
        "source_pattern": pattern,
        "source_files": source_files,
        "input_rows": len(source_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_categories": count_by(train_rows, "category"),
        "val_categories": count_by(val_rows, "category"),
        "train_persona_strength": count_by_persona(train_rows),
        "val_persona_strength": count_by_persona(val_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the workspace root data files into JSONL train/val files for QLoRA training.")
    parser.add_argument("--source_dir", type=str, default=str(default_source_dir()))
    parser.add_argument("--source_pattern", type=str, default="json*.json")
    parser.add_argument("--train_out", type=str, default="data/generated/custom_train.jsonl")
    parser.add_argument("--val_out", type=str, default="data/generated/custom_val.jsonl")
    parser.add_argument("--manifest_out", type=str, default="data/generated/custom_data_manifest.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_dir = Path(args.source_dir).expanduser().resolve()
    train_out = (project_root / args.train_out).resolve() if not Path(args.train_out).is_absolute() else Path(args.train_out)
    val_out = (project_root / args.val_out).resolve() if not Path(args.val_out).is_absolute() else Path(args.val_out)
    manifest_out = (project_root / args.manifest_out).resolve() if not Path(args.manifest_out).is_absolute() else Path(args.manifest_out)

    source_rows = load_rows(source_dir, args.source_pattern)
    train_rows, val_rows = split_rows(source_rows, val_ratio=args.val_ratio, seed=args.seed)

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(
            build_manifest(
                source_dir=source_dir,
                pattern=args.source_pattern,
                train_rows=train_rows,
                val_rows=val_rows,
                source_rows=source_rows,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(train_rows)} train rows to {train_out}")
    print(f"Wrote {len(val_rows)} val rows to {val_out}")
    print(f"Wrote manifest to {manifest_out}")


if __name__ == "__main__":
    main()
