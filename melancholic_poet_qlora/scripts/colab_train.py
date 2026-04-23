from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_drive_if_requested() -> None:
    from google.colab import drive

    drive.mount("/content/drive")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def run_command(args: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"$ {' '.join(args)}")
    subprocess.run(args, cwd=str(cwd), env=env, check=True)


def resolve_output_dir(config_path: Path, root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["training"]["output_dir"])
    return output_dir if output_dir.is_absolute() else (root / output_dir).resolve()


def copy_outputs_to_drive(output_dir: Path, drive_output_dir: Path) -> Path:
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    destination = drive_output_dir / output_dir.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(output_dir, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare root data and launch training from Google Colab without adding Colab as a hard dependency.")
    parser.add_argument("--config", type=str, default="configs/custom_qwen25_15b.yaml")
    parser.add_argument("--source_dir", type=str, default=None, help="Override the root data directory used by prepare_root_data.py.")
    parser.add_argument("--source_pattern", type=str, default="json*.json")
    parser.add_argument("--train_out", type=str, default="data/generated/custom_train.jsonl")
    parser.add_argument("--val_out", type=str, default="data/generated/custom_val.jsonl")
    parser.add_argument("--manifest_out", type=str, default="data/generated/custom_data_manifest.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip_prepare_data", action="store_true")
    parser.add_argument("--mount_drive", action="store_true")
    parser.add_argument("--drive_output_dir", type=str, default=None, help="Optional mounted-Drive directory where the run output directory will be copied after training.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--set", action="append", default=[], help="Forwarded config overrides for train.py, e.g. training.max_steps=80")
    args = parser.parse_args()

    root = project_root()
    env = build_env(root)

    if args.mount_drive:
        if not running_in_colab():
            raise EnvironmentError("--mount_drive was requested, but this process is not running inside Google Colab.")
        mount_drive_if_requested()

    if not args.skip_prepare_data:
        prepare_cmd = [sys.executable, "scripts/prepare_root_data.py"]
        if args.source_dir:
            prepare_cmd.extend(["--source_dir", args.source_dir])
        prepare_cmd.extend(
            [
                "--source_pattern",
                args.source_pattern,
                "--train_out",
                args.train_out,
                "--val_out",
                args.val_out,
                "--manifest_out",
                args.manifest_out,
                "--val_ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
            ]
        )
        run_command(prepare_cmd, cwd=root, env=env)

    train_cmd = [sys.executable, "train.py", "--config", args.config]
    if args.resume_from_checkpoint:
        train_cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    for override in args.set:
        train_cmd.extend(["--set", override])
    run_command(train_cmd, cwd=root, env=env)

    if args.drive_output_dir:
        if not running_in_colab():
            raise EnvironmentError("--drive_output_dir was provided, but this process is not running inside Google Colab.")
        output_dir = resolve_output_dir((root / args.config).resolve(), root=root)
        destination = copy_outputs_to_drive(output_dir=output_dir, drive_output_dir=Path(args.drive_output_dir).expanduser())
        print(f"Copied outputs to {destination}")


if __name__ == "__main__":
    main()
