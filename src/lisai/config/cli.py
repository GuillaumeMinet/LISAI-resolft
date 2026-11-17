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
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.resources import BUILTIN_NAMESPACES, find_resource, iter_resource_files
from lisai.config.settings import Settings, load_default_settings
from lisai.config.io.yaml import load_yaml
from lisai.preprocess.templates import create_preprocess_config_from_template
from lisai.config.user_configs import refresh_user_schemas


def _is_builtin_training_name(ref: str) -> bool:
    from pathlib import Path

    path = Path(ref)
    return (not path.is_absolute() and len(path.parts) >= 2
            and path.parts[0] in BUILTIN_NAMESPACES["training"])


def _builtin_training_reference(ref: str):
    """Read-only package references need no workspace or Settings object."""
    from pathlib import Path

    if not _is_builtin_training_name(ref):
        return None
    path = Path(ref)
    return find_resource("training", path.parts[0], Path(*path.parts[1:]))


def run_list_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, settings: Settings | None = None,
) -> int:
    domain = getattr(args, "domain", "training")
    namespace = {"template": "templates", "preset": "presets", "example": "examples"}.get(args.kind)
    if args.kind and namespace is not None and namespace not in BUILTIN_NAMESPACES.get(domain, ()):
        parser.error(f"{args.kind!r} is not available for {domain} configurations")
    if args.task and domain != "training":
        parser.error("--task only applies to training configs.")

    # A specifically scoped built-in listing never accesses workspace settings.
    stg = settings
    if stg is None and namespace is None:
        stg = load_default_settings()
    if domain == "training":
        entries = discover_training_configs(kind=args.kind, task=args.task, stg=stg)
        if not entries:
            parser.exit(status=1, message="No training configs found.\n")
        print("kind      owner      task                    config")
        print("--------  ---------  ----------------------  -------------------------------")
        for entry in entries:
            task = entry.task or "-"
            print(f"{entry.kind:<8}  {entry.ownership:<9}  {task:<22}  {entry.relative_path}")
            if args.verbose and entry.description:
                print(f"          {entry.description}")
        return 0

    builtin_entries = []
    if namespace is not None or args.kind is None:
        for ns in ([namespace] if namespace else BUILTIN_NAMESPACES[domain]):
            builtin_entries.extend(("built-in", f"{ns}/{name}")
                                   for name, _ in iter_resource_files(domain, ns))
    user_entries = []
    if namespace is None:
        assert stg is not None
        user_entries = [("local", name) for name, _ in
                        ConfigPathResolver(domain, stg=stg).user_config_paths()]
    entries = sorted(builtin_entries + user_entries, key=lambda e: e[1])
    if not entries:
        parser.exit(status=1, message=f"No {domain} configs found.\n")
    print("owner      config")
    print("---------  --------------------------------")
    for owner, name in entries:
        print(f"{owner:<9}  {name}")
    return 0


def run_info_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, settings: Settings | None = None,
) -> int:
    package = _builtin_training_reference(args.config)
    if _is_builtin_training_name(args.config):
        if package is None:
            parser.exit(status=1, message=f"Packaged training config not found: {args.config}\n")
        from lisai.config.io.metadata import split_config_metadata
        body, metadata = split_config_metadata(load_yaml(package))
        path = package
        ownership = "built-in"
    else:
        stg = load_default_settings() if settings is None else settings
        try:
            body, metadata, path = load_training_config(args.config, stg=stg)
        except Exception as exc:
            parser.exit(status=1, message=f"{exc}\n")
        source = ConfigPathResolver("training", stg=stg).resolve_reference(args.config)
        ownership = source.ownership if source is not None else "unknown"

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
    print(f"owner: {ownership}")
    print(f"kind: {metadata.get('kind', '-')}")
    print(f"name: {metadata.get('name', path.stem)}")
    print(f"task: {metadata.get('task', task or '-')}")
    description = metadata.get("description")
    if description:
        print(f"description: {description}")
    return 0


def run_validate_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, settings: Settings | None = None,
) -> int:
    package = _builtin_training_reference(args.config)
    if _is_builtin_training_name(args.config) and package is None:
        parser.exit(status=1, message=f"Packaged training config not found: {args.config}\n")
    stg = settings
    try:
        if package is not None:
            path = package
            owner = "built-in"
        else:
            stg = load_default_settings() if stg is None else stg
            path = resolve_training_config_path(args.config, stg=stg)
            owner = ConfigPathResolver("training", stg=stg).resolve_reference(args.config).ownership
        cfg = load_yaml(path)
        if args.template:
            validate_training_template_dict(cfg)
        else:
            validate_training_config_dict(cfg)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    mode = "template" if args.template else "training config"
    print(f"Valid {mode} ({owner}): {path}")
    return 0


def run_resolve_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, settings: Settings | None = None,
) -> int:
    stg = load_default_settings() if settings is None else settings
    try:
        _, _, path = load_training_config(args.config, stg=stg)
        cfg = resolve_config_dict(load_yaml(path), stg=stg)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    source = ConfigPathResolver("training", stg=stg).resolve_reference(args.config)
    print(f"# source: {source.ownership if source else 'unknown'}")
    print(yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), sort_keys=False))
    return 0


def run_new_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, settings: Settings | None = None,
) -> int:
    stg = load_default_settings() if settings is None else settings
    if args.preset == "preprocess":
        if not args.preprocess_template:
            parser.error("Specify a preprocess template (e.g. single_recon).")
        forbidden = [
            "--custom" if args.custom else None,
            "--name" if args.name else None,
            "--input" if args.input is not None else None,
            "--target" if args.target is not None else None,
            "--betaKL" if args.betaKL is not None else None,
            "--loss" if args.loss is not None else None,
            "--upsampling-factor" if args.upsampling_factor is not None else None,
            "--sampling-ratio" if args.sampling_ratio is not None else None,
            "--temporal-window" if args.temporal_window is not None else None,
        ]
        if any(forbidden):
            parser.error("Preprocess creation does not accept training flags: " +
                         ", ".join(flag for flag in forbidden if flag))
        try:
            output_path, warnings = create_preprocess_config_from_template(
                args.preprocess_template, dataset_name=args.dataset,
                output=args.output, overwrite=args.overwrite, stg=stg,
            )
        except (FileExistsError, FileNotFoundError, ValueError, ValidationError) as exc:
            parser.exit(status=1, message=f"{exc}\n")
        print(f"Created preprocess config: {output_path}  [local]")
        print(f"Source: built-in preprocess template {args.preprocess_template}")
        for warning in warnings:
            print(f"Warning: {warning}")
        return 0
    if args.preprocess_template:
        parser.error("The second positional argument is only valid for `new preprocess`.")
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
                stg=stg,
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
                stg=stg,
            )
    except (FileExistsError, FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Created training config: {output_path}  [local]")
    print(f"Source: built-in {'template' if args.custom else 'preset'}")
    for warning in warnings:
        print(f"Warning: {warning}")
    return 0


def _selected_settings(args: argparse.Namespace, settings: Settings | None) -> Settings | None:
    """Use the single settings object injected by the top-level CLI, if present."""
    return settings if settings is not None else getattr(args, "_lisai_settings", None)


def _add_config_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    config_subparsers = parser.add_subparsers(dest="configs_command")
    config_subparsers.required = True

    list_parser = config_subparsers.add_parser(
        "list",
        help="List packaged and user-managed experiment configs.",
        description="List package resources or user-managed configs from the selected workspace.",
    )
    list_parser.add_argument("--domain", choices=["training", "preprocess", "inference"], default="training",
                             help="List training configs, preprocess templates, or inference configs.")
    list_parser.add_argument("--kind", choices=["preset", "template", "example", "local"])
    list_parser.add_argument("--task")
    list_parser.add_argument("--verbose", action="store_true")
    list_parser.set_defaults(handler=lambda args, p=list_parser, settings=None: run_list_from_args(args, p, settings=_selected_settings(args, settings)))

    info_parser = config_subparsers.add_parser(
        "info",
        help="Show metadata and inferred task information for a training config.",
        description="Show metadata and inferred task information for a training config.",
    )
    info_parser.add_argument("config", help="Namespaced package resource, managed training name, or explicit file path.")
    info_parser.set_defaults(handler=lambda args, p=info_parser, settings=None: run_info_from_args(args, p, settings=_selected_settings(args, settings)))

    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate a training config or template.",
        description="Validate a training config or template.",
    )
    validate_parser.add_argument("config", help="Namespaced package resource, managed training name, or explicit file path.")
    validate_parser.add_argument("--template", action="store_true", help="Use relaxed template validation.")
    validate_parser.set_defaults(handler=lambda args, p=validate_parser, settings=None: run_validate_from_args(args, p, settings=_selected_settings(args, settings)))

    resolve_parser = config_subparsers.add_parser(
        "resolve",
        help="Print the fully resolved training config.",
        description="Print the fully resolved training config.",
    )
    resolve_parser.add_argument("config", help="Namespaced package resource, managed training name, or explicit file path.")
    resolve_parser.set_defaults(handler=lambda args, p=resolve_parser, settings=None: run_resolve_from_args(args, p, settings=_selected_settings(args, settings)))

    new_parser = config_subparsers.add_parser(
        "new",
        help="Create training or preprocess YAMLs from packaged presets/templates.",
        description="Create an editable YAML in the selected flat writable configuration root.",
    )
    new_parser.add_argument(
        "preset",
        nargs="?",
        help="Training preset, or `preprocess` followed by a preprocess template name.",
    )
    new_parser.add_argument("preprocess_template", nargs="?", help="Template name after `preprocess`.")
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
        help="Input subfolder for data.input. Use --input or --input= for an empty string.",
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
    new_parser.add_argument("--output", help="Managed name under training/ or preprocess/; ./, ../ or absolute for an explicit path.")
    new_parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output config.")
    new_parser.set_defaults(handler=lambda args, p=new_parser, settings=None: run_new_from_args(args, p, settings=_selected_settings(args, settings)))
    schema_parser = config_subparsers.add_parser(
        "schema", help="Explicitly manage editor JSON schemas under the selected config root."
    )
    schema_commands = schema_parser.add_subparsers(dest="schema_command", required=True)
    refresh_parser = schema_commands.add_parser(
        "refresh", help="Regenerate workspace-aware editor JSON schemas."
    )
    refresh_parser.set_defaults(
        handler=lambda args, p=refresh_parser, settings=None:
            run_schema_refresh_from_args(args, p, settings=_selected_settings(args, settings))
    )
    return parser


def run_schema_refresh_from_args(args, parser, *, settings: Settings | None = None) -> int:
    stg = load_default_settings() if settings is None else settings
    try:
        generated = refresh_user_schemas(stg=stg)
        # The developer selector always lives in the verified checkout, even
        # when config.mode selects a workspace or custom experiment root.
        # Refresh its schema there too, rather than leaving an old schema
        # behind after switching config roots.
        locations = getattr(args, "_lisai_locations", None)
        local = (locations.local_config_path if locations is not None and
                 locations.developer_mode else None)
        if local is not None:
            from lisai.config.json_schema import write_local_config_json_schema
            developer_schema = local.parent / "schema" / "local-config.schema.json"
            if developer_schema not in generated:
                generated.append(write_local_config_json_schema(developer_schema))
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    print(f"Refreshed {len(generated)} editor schemas in {stg.CONFIGS_ROOT / 'schema'}")
    if local is not None and developer_schema != stg.CONFIGS_ROOT / "schema/local-config.schema.json":
        print(f"Developer selector schema: {developer_schema}")
    return 0


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


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args, settings=settings)


__all__ = [
    "TrainingConfigKind",
    "TrainingPresetName",
    "add_configs_subparser",
    "main",
]
