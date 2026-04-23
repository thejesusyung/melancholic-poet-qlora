from __future__ import annotations

import argparse

from poetune.public_domain import build_public_domain_augmentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn curated public-domain snippets into optional style-augmentation SFT rows.")
    parser.add_argument("--raw_dir", type=str, default="data/public_domain/raw")
    parser.add_argument("--out_path", type=str, default="data/generated/public_domain_aug.jsonl")
    parser.add_argument("--max_examples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    count = build_public_domain_augmentation(
        raw_dir=args.raw_dir,
        out_path=args.out_path,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    print(f"Wrote {count} rows to {args.out_path}")


if __name__ == "__main__":
    main()
