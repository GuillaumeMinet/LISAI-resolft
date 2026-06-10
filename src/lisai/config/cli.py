from __future__ import annotations

import argparse
from typing import Sequence

import yaml
from pydantic import ValidationError

from lisai.config.catalog import (
    TrainingConfigKind,
    TrainingPresetName,
    create_training_config_from_preset,
    create_training_config_from_template,
    discover_training_configs,
    load_training_config,
    resolve_training_config_path,
    validate_training_config_dict,
    validate_training_template_dict,
)
from lisai.config.io.resolver import resolve_config_dict
from lisai.config.io.yaml import load_yaml


def run_list_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    entries = discover_training_configs(kind=args.kind, task=args.task)
    if not entries:
        parser.exit(status=1, message="No training configs found.\n")

    print("kind      task                    config")
    print("--------  ----------------------  -------------------------------")
    for entry in entries:
        task = entry.task or "-"
        print(f"{entry.kind:<8}  {task:<22}  {entry.relative_path}")
        if args.verbose and entry.description:
            print(f"          {entry.description}")
    return 0


def run_info_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        body, metadata, path = load_training_config(args.config)
    except Exception as exc:
        parser.exit(status=1, message=f"{exc}\n")

    experiment = body.get("experiment") if isinstance(body, dict) else None
    task = None
    if isinstance(experiment, dict):
        task_cfg = experiment.get("task")
        if isinstance(task_cfg, dict):
            task = task_cfg.get("name")
        elif isinstance(task_cfg, str):
            task = task_cfg

    metadata = metadata or {}
    print(f"path: {path}")
    print(f"kind: {metadata.get('kind', '-')}")
    print(f"name: {metadata.get('name', path.stem)}")
    print(f"task: {metadata.get('task', task or '-')}")
    description = metadata.get("description")
    if description:
        print(f"description: {description}")
    return 0


def run_validate_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        path = resolve_training_config_path(args.config)
        cfg = load_yaml(path)
        if args.template:
            validate_training_template_dict(cfg)
        else:
            validate_training_config_dict(cfg)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    mode = "template" if args.template else "training config"
    print(f"Valid {mode}: {path}")
    return 0


def run_resolve_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        _, _, path = load_training_config(args.config)
        cfg = resolve_config_dict(load_yaml(path))
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), sort_keys=False))
    return 0


def run_new_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.custom and args.preset:
        parser.error("Choose either a preset name or --custom, not both.")
    if not args.custom and not args.preset:
        parser.error("Choose a preset name or use --custom.")
    if args.custom:
        preset_only = {
            "--betaKL": args.betaKL,
            "--loss": args.loss,
            "--upsampling-factor": args.upsampling_factor,
            "--sampling-ratio": args.sampling_ratio,
            "--temporal-window": args.temporal_window,
        }
        used_preset_only = [flag for flag, value in preset_only.items() if value is not None]
        if used_preset_only:
            joined = ", ".join(used_preset_only)
            parser.error(f"{joined} can only be used with presets.")

    try:
        if args.custom:
            output_path, warnings = create_training_config_from_template(
                exp_name=args.name,
                dataset_name=args.dataset,
                input_name=args.input,
                target_name=args.target,
                output=args.output,
                overwrite=args.overwrite,
            )
        else:
            output_path, warnings = create_training_config_from_preset(
                args.preset,
                exp_name=args.name,
                dataset_name=args.dataset,
                input_name=args.input,
                target_name=args.target,
                betaKL=args.betaKL,
                loss=args.loss,
                upsampling_factor=args.upsampling_factor,
                sampling_ratio=args.sampling_ratio,
                temporal_window=args.temporal_window,
                output=args.output,
                overwrite=args.overwrite,
            )
    except (FileExistsError, FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Created training config: {output_path}")
    for warning in warnings:
        print(f"Warning: {warning}")
    return 0


def _add_config_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    config_subparsers = parser.add_subparsers(dest="configs_command")
    config_subparsers.required = True

    list_parser = config_subparsers.add_parser(
        "list",
        help="List tracked and local training configs.",
        description="List tracked and local training configs.",
    )
    list_parser.add_argument("--kind", choices=["preset", "template", "example", "local"])
    list_parser.add_argument("--task")
    list_parser.add_argument("--verbose", action="store_true")
    list_parser.set_defaults(handler=lambda args, p=list_parser: run_list_from_args(args, p))

    info_parser = config_subparsers.add_parser(
        "info",
        help="Show metadata and inferred task information for a training config.",
        description="Show metadata and inferred task information for a training config.",
    )
    info_parser.add_argument("config", help="Explicit path or configs/training-relative subpath.")
    info_parser.set_defaults(handler=lambda args, p=info_parser: run_info_from_args(args, p))

    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate a training config or template.",
        description="Validate a training config or template.",
    )
    validate_parser.add_argument("config", help="Explicit path or configs/training-relative subpath.")
    validate_parser.add_argument("--template", action="store_true", help="Use relaxed template validation.")
    validate_parser.set_defaults(handler=lambda args, p=validate_parser: run_validate_from_args(args, p))

    resolve_parser = config_subparsers.add_parser(
        "resolve",
        help="Print the fully resolved training config.",
        description="Print the fully resolved training config.",
    )
    resolve_parser.add_argument("config", help="Explicit path or configs/training-relative subpath.")
    resolve_parser.set_defaults(handler=lambda args, p=resolve_parser: run_resolve_from_args(args, p))

    new_parser = config_subparsers.add_parser(
        "new",
        help="Create a local training config from a preset or template.",
        description="Create a local training config from a preset or template.",
    )
    new_parser.add_argument(
        "preset",
        nargs="?",
        help="Preset to instantiate, such as denoising_hdn_unsup or upsamp_single_frame.",
    )
    new_parser.add_argument(
        "--custom",
        action="store_true",
        help="Start from templates/base_training.yml instead of a preset.",
    )
    new_parser.add_argument("--name", help="Experiment name for experiment.exp_name. Defaults to CHANGEME.")
    new_parser.add_argument("--dataset", help="Dataset name for data.dataset_name. Defaults to CHANGEME.")
    new_parser.add_argument(
        "--input",
        nargs="?",
        const="",
        default=None,
        help="Input subfolder or input key for data.input. Use --input or --input= for an empty string.",
    )
    new_parser.add_argument("--target", help="Target subfolder or key for paired datasets.")
    new_parser.add_argument("--betaKL", "--beta-kl", dest="betaKL", type=float)
    new_parser.add_argument(
        "--loss",
        choices=["MSE", "mse", "l2", "L2", "MAE", "mae", "l1", "L1", "CharEdge_loss", "CharEdge"],
    )
    new_parser.add_argument("--upsampling-factor", type=int)
    new_parser.add_argument("--sampling-ratio", type=float, choices=[0.25, 0.5, 0.75])
    new_parser.add_argument("--temporal-window", type=int)
    new_parser.add_argument("--output", help="Output path. Defaults to configs/training/local/<name>.yml.")
    new_parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output config.")
    new_parser.set_defaults(handler=lambda args, p=new_parser: run_new_from_args(args, p))
    return parser


def add_configs_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "configs",
        help="Discover, validate, resolve, and create LISAI configs.",
        description="Discover, validate, resolve, and create LISAI configs.",
    )
    _add_config_commands(parser)
    return parser


def build_parser(*, prog: str = "lisai configs") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover, validate, resolve, and create LISAI configs.", prog=prog)
    return _add_config_commands(parser)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = [
    "TrainingConfigKind",
    "TrainingPresetName",
    "add_configs_subparser",
    "main",
]
