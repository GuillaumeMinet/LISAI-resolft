from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lisai.config.io.config_paths import ConfigPathResolver

from .run_training import run_training

training_config_paths = ConfigPathResolver("training")


def resolve_config_path(config_arg: str) -> Path:
    resolved = training_config_paths.resolve(config_arg)
    assert resolved is not None
    _ensure_trainable_config_path(resolved)
    return resolved


def _ensure_trainable_config_path(path: Path) -> None:
    root = training_config_paths.root.resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return
    if relative.parts and relative.parts[0] in {"presets", "templates"}:
        raise ValueError(
            "Training presets and templates must be instantiated before training. "
            "Use `lisai configs new ...` to create a local training config."
        )


def _get_config_arg(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    positional = getattr(args, "config", None)
    option = getattr(args, "config_option", None)

    if positional and option:
        parser.error("Provide the training config either positionally or via --config, not both.")
    if positional:
        return positional
    if option:
        return option

    parser.error("A training config is required.")
    raise AssertionError("argparse.error should raise SystemExit")


def run_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config_arg = _get_config_arg(args, parser)
    try:
        config_path = resolve_config_path(config_arg)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    run_training(config_path)
    return 0


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "config",
        nargs="?",
        help=f"Path to a YAML config file, or a config name from {training_config_paths.root} with or without .yml/.yaml.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help=f"Path to a YAML config file, or a config name from {training_config_paths.root} with or without .yml/.yaml.",
    )
    return parser


def build_parser(*, prog: str = "lisai train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model using a YAML config", prog=prog)
    add_train_arguments(parser)
    return parser


def add_train_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "train",
        help="Train a model from a YAML config.",
        description="Train a model using a YAML config",
    )
    add_train_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_from_args(args, p))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_from_args(args, parser)
