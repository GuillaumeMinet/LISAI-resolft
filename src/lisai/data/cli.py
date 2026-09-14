from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.runs.cli import _try_open_path

from .dataset_registry import load_dataset_registry
from .readme import dataset_readme_path, ensure_dataset_readme
from .rename import DatasetRenameError, apply_dataset_rename, build_dataset_rename_plan


DESCRIPTION_PREVIEW_MAX_CHARS = 40


def _paths() -> Paths:
    return Paths(settings)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _format_path_value(value: Any) -> str:
    if value is None:
        return "null"
    if value == "":
        return "<root>"
    return str(value)


def _format_range(value: Any) -> str:
    if isinstance(value, Mapping):
        minimum = value.get("min")
        maximum = value.get("max")
        if minimum is None or maximum is None:
            return "-"
        if minimum == maximum:
            return str(minimum)
        return f"{minimum}-{maximum}"
    if isinstance(value, list):
        if not value:
            return "-"
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            minimum = min(value)
            maximum = max(value)
            return str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
    if value is None:
        return "-"
    return str(value)


def _data_types(info: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    for section_name in ("size", "split", "structure", "outputs", "defaults"):
        section = info.get(section_name)
        if isinstance(section, Mapping):
            names.update(str(key) for key in section)
    return sorted(names)


def _format_files(info: Mapping[str, Any], data_types: Sequence[str]) -> str:
    size = _as_mapping(info.get("size"))
    values: list[str] = []
    for data_type in data_types:
        size_entry = _as_mapping(size.get(data_type))
        n_files = size_entry.get("n_files")
        if n_files is None:
            continue
        values.append(str(n_files) if len(data_types) == 1 else f"{data_type}:{n_files}")
    return ",".join(values) if values else "-"


def _format_frames(info: Mapping[str, Any], data_types: Sequence[str]) -> str:
    size = _as_mapping(info.get("size"))
    data_format = info.get("data_format")
    values: list[str] = []
    for data_type in data_types:
        size_entry = _as_mapping(size.get(data_type))
        n_frames = size_entry.get("n_frames")
        if n_frames is None and data_format == "single":
            n_frames = size_entry.get("n_files")
        if n_frames is None:
            continue
        values.append(str(n_frames) if len(data_types) == 1 else f"{data_type}:{n_frames}")
    return ",".join(values) if values else "-"


def _format_split(info: Mapping[str, Any], data_types: Sequence[str]) -> str:
    split = _as_mapping(info.get("split"))
    values: list[str] = []
    for data_type in data_types:
        counts = _as_mapping(_as_mapping(split.get(data_type)).get("counts"))
        if not counts:
            continue
        summary = f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('test', 0)}"
        values.append(summary if len(data_types) == 1 else f"{data_type}:{summary}")
    return ",".join(values) if values else "-"


def _format_range_summary(info: Mapping[str, Any], data_types: Sequence[str]) -> str:
    size = _as_mapping(info.get("size"))
    values: list[str] = []
    for data_type in data_types:
        size_entry = _as_mapping(size.get(data_type))
        if "timepoints" in size_entry:
            summary = f"t={_format_range(size_entry.get('timepoints'))}"
        elif "snr_levels" in size_entry:
            summary = f"snr={_format_range(size_entry.get('snr_levels'))}"
        else:
            continue
        values.append(summary if len(data_types) == 1 else f"{data_type}:{summary}")
    return ",".join(values) if values else "-"


def _format_defaults(info: Mapping[str, Any], data_types: Sequence[str]) -> str:
    defaults = _as_mapping(info.get("defaults"))
    values: list[str] = []
    labels = {"input": "in", "target": "target", "eval_gt": "gt"}
    for data_type in data_types:
        defaults_entry = _as_mapping(defaults.get(data_type))
        tokens = [
            f"{label}={_format_path_value(defaults_entry[key])}"
            for key, label in labels.items()
            if defaults_entry.get(key) is not None
        ]
        if not tokens:
            continue
        summary = ",".join(tokens)
        values.append(summary if len(data_types) == 1 else f"{data_type}:{summary}")
    return ";".join(values) if values else "-"


def _format_description(value: Any, *, full: bool) -> str:
    if value is None:
        return "-"
    text = " ".join(str(value).split())
    if not text:
        return "-"
    if full or len(text) <= DESCRIPTION_PREVIEW_MAX_CHARS:
        return text
    keep = DESCRIPTION_PREVIEW_MAX_CHARS - 3
    return text[:keep].rstrip() + "..."


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _registry_from_paths(paths: Paths) -> dict[str, dict[str, Any]]:
    return load_dataset_registry(paths.dataset_registry_path())


def _usage_sort_key(item: tuple[str, Mapping[str, Any]]) -> tuple[int, str]:
    name, info = item
    usage = str(info.get("usage") or "").lower()
    order = {"training": 0, "evaluation": 1}
    return order.get(usage, 2), name.casefold()


def list_datasets(
    *,
    paths: Paths | None = None,
    usage: str | None = None,
    full_description: bool = False,
    include_description: bool = False,
) -> None:
    paths = paths or _paths()
    registry = _registry_from_paths(paths)
    if usage is not None:
        registry = {
            name: info
            for name, info in registry.items()
            if str(info.get("usage") or "").lower() == usage
        }
    if not registry:
        print(f"No datasets found in {paths.dataset_registry_path()}")
        return

    headers = (
        "name", "usage", "format", "types", "files", "frames", "split", "range", "defaults",
    )
    if include_description:
        headers += ("description",)

    rows: list[tuple[str, ...]] = []
    for name, info in sorted(registry.items(), key=_usage_sort_key):
        data_types = _data_types(info)
        row = (
            name,
            str(info.get("usage") or "-"),
            str(info.get("data_format") or "-"),
            ",".join(data_types) if data_types else "-",
            _format_files(info, data_types),
            _format_frames(info, data_types),
            _format_split(info, data_types),
            _format_range_summary(info, data_types),
            _format_defaults(info, data_types),
        )

        if include_description:
            row+=(
                _format_description(
                    info.get("description"),
                    full=full_description,
                ),
            )
        rows.append(row)
    print(_table(headers, rows))


def _require_dataset(
    name: str,
    *,
    paths: Paths,
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, Any], Path]:
    registry = _registry_from_paths(paths)
    info = registry.get(name)
    if info is None:
        known = ", ".join(sorted(registry)) or "none"
        parser.exit(status=1, message=f"Unknown dataset {name!r}. Known datasets: {known}\n")
    usage = str(info.get("usage") or "training").lower()
    return info, paths.dataset_dir(dataset_name=name, usage=usage)


def show_dataset(name: str, *, paths: Paths | None = None, parser: argparse.ArgumentParser) -> None:
    paths = paths or _paths()
    info, dataset_dir = _require_dataset(name, paths=paths, parser=parser)

    print(f"Dataset: {name}")
    print(f"Path: {dataset_dir.resolve()}")
    print(f"Usage: {info.get('usage') or '-'}")
    print(f"Format: {info.get('data_format') or '-'}")
    print(f"Description: {info.get('description') or '-'}")

    data_types = _data_types(info)
    size = _as_mapping(info.get("size"))
    split = _as_mapping(info.get("split"))
    outputs = _as_mapping(info.get("outputs"))
    defaults = _as_mapping(info.get("defaults"))

    for data_type in data_types:
        print("")
        print(f"{data_type}:")

        size_entry = _as_mapping(size.get(data_type))
        if size_entry:
            for key, value in size_entry.items():
                rendered = _format_range(value) if key in {"timepoints", "snr_levels"} else value
                print(f"  {key}: {rendered}")

        counts = _as_mapping(_as_mapping(split.get(data_type)).get("counts"))
        if counts:
            print(
                "  split: "
                f"train={counts.get('train', 0)} "
                f"val={counts.get('val', 0)} "
                f"test={counts.get('test', 0)}"
            )

        output_entries = outputs.get(data_type)
        if isinstance(output_entries, list) and output_entries:
            print("  outputs:")
            output_rows = [
                (
                    str(entry.get("key", "-")),
                    _format_path_value(entry.get("path")),
                    str(entry.get("role", "-")),
                    str(entry.get("axes", "-")),
                )
                for entry in output_entries
                if isinstance(entry, Mapping)
            ]
            if output_rows:
                for line in _table(("key", "path", "role", "axes"), output_rows).splitlines():
                    print(f"    {line}")

        defaults_entry = _as_mapping(defaults.get(data_type))
        if defaults_entry:
            print("  defaults:")
            for key in ("input", "target", "eval_gt"):
                print(f"    {key}: {_format_path_value(defaults_entry.get(key))}")

    readme_path = dataset_readme_path(dataset_dir)
    print("")
    if not readme_path.exists():
        print("README: not provided")
    else:
        print("README:")
        content = readme_path.read_text(encoding="utf-8")
        if content:
            print(content.rstrip("\n"))


def open_dataset(name: str, *, paths: Paths | None = None, parser: argparse.ArgumentParser) -> int:
    paths = paths or _paths()
    _, dataset_dir = _require_dataset(name, paths=paths, parser=parser)
    resolved = dataset_dir.resolve()
    if _try_open_path(resolved):
        return 0
    print(resolved)
    return 0


def open_dataset_readme(
    name: str,
    *,
    paths: Paths | None = None,
    parser: argparse.ArgumentParser,
) -> int:
    paths = paths or _paths()
    _, dataset_dir = _require_dataset(name, paths=paths, parser=parser)
    if not dataset_dir.exists():
        parser.exit(
            status=1,
            message=f"Dataset directory does not exist: {dataset_dir.resolve()}\n",
        )

    readme_path, _ = ensure_dataset_readme(dataset_dir)
    resolved = readme_path.resolve()
    if _try_open_path(resolved):
        return 0
    print(resolved)
    return 0


def _print_rename_plan(plan) -> None:
    print("Dataset rename")
    print("")
    print(f"  {plan.old_name}  ->  {plan.new_name}")
    print(f"  Usage: {plan.usage}")
    print("")
    print("This will change:")
    print("  - dataset folder")
    print(f"    {plan.source_dir}")
    print(f"    -> {plan.destination_dir}")
    print("")
    print("  - dataset registry entry")
    print(f"    {plan.old_name} -> {plan.new_name}")
    if plan.usage == "training":
        print("")
        print(f"  - {plan.active_run_metadata_count} active run metadata file(s)")
        print(f"  - {plan.training_config_count} saved training config(s)")
        print(f"  - {plan.split_manifest_count} split manifest(s)")
        print(f"  - {plan.archived_run_metadata_count} archived run metadata file(s)")
    print("")
    print("Not changed:")
    print("  - TensorBoard logs")
    print("    Existing logs remain under the old dataset name; future logs may be split")
    print("    between the old and new dataset folders.")
    print("  - promoted/exported model packages and historical evaluation provenance")
    print("  - arbitrary external configs or references outside LISAI-managed dataset metadata")


def _confirm_dataset_rename() -> bool:
    print("")
    print("Proceed with dataset rename? [y/N] ", end="", flush=True)
    answer = sys.stdin.readline()
    return answer.strip().casefold() in {"y", "yes"}


def rename_dataset(
    old_name: str,
    new_name: str,
    *,
    paths: Paths | None = None,
) -> int:
    paths = paths or _paths()
    try:
        plan = build_dataset_rename_plan(old_name, new_name, paths=paths)
    except DatasetRenameError as exc:
        print(f"Cannot rename dataset: {exc}")
        return 1

    _print_rename_plan(plan)
    if not _confirm_dataset_rename():
        print("Dataset rename cancelled.")
        return 0

    try:
        apply_dataset_rename(plan)
    except DatasetRenameError as exc:
        print(f"Dataset rename failed: {exc}")
        return 1

    print("")
    print(f"Renamed dataset: {plan.old_name} -> {plan.new_name}")
    print(f"Updated folder and registry ({plan.usage}).")
    if plan.usage == "training":
        print(
            "Updated "
            f"{plan.active_run_metadata_count} active run metadata file(s), "
            f"{plan.training_config_count} training config(s), "
            f"{plan.split_manifest_count} split manifest(s), and "
            f"{plan.archived_run_metadata_count} archived run metadata file(s)."
        )
    return 0


def run_list_from_args(args: argparse.Namespace) -> int:
    usage = None
    if args.training:
        usage = "training"
    elif args.evaluation:
        usage = "evaluation"
    list_datasets(usage=usage, 
                  full_description=args.full,
                  include_description=not args.short)
    return 0


def _add_dataset_list_arguments(parser: argparse.ArgumentParser) -> None:
    usage_group = parser.add_mutually_exclusive_group()
    usage_group.add_argument(
        "--training",
        action="store_true",
        help="Show only training datasets.",
    )
    usage_group.add_argument(
        "--evaluation",
        action="store_true",
        help="Show only evaluation datasets.",
    )

    description_group = parser.add_mutually_exclusive_group()

    description_group.add_argument(
        "--full",
        action="store_true",
        help="Show full dataset descriptions instead of compact previews.",
    )

    description_group.add_argument(
        "--short",
        action="store_true",
        help="Hide dataset descriptions.",
    )


def run_show_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    show_dataset(args.name, parser=parser)
    return 0


def run_open_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    return open_dataset(args.name, parser=parser)


def run_open_readme_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    return open_dataset_readme(args.name, parser=parser)


def run_rename_from_args(args: argparse.Namespace) -> int:
    return rename_dataset(args.old_name, args.new_name)


def _add_dataset_rename_parser(subparsers):
    rename_parser = subparsers.add_parser(
        "rename",
        help="Rename a registered dataset and update LISAI-managed metadata.",
        description=(
            "Rename a registered dataset folder and registry key, updating affected "
            "training-run metadata. This is an impactful operation and always requires "
            "interactive confirmation after a preflight summary."
        ),
    )
    rename_parser.add_argument("old_name", help="Current dataset name from the registry.")
    rename_parser.add_argument("new_name", help="New dataset name.")
    rename_parser.set_defaults(handler=run_rename_from_args)
    return rename_parser


def add_datasets_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "datasets",
        help="Inspect registered LISAI datasets.",
        description="Inspect registered LISAI datasets.",
    )
    dataset_subparsers = parser.add_subparsers(dest="datasets_command")
    dataset_subparsers.required = True

    list_parser = dataset_subparsers.add_parser(
        "list",
        help="List registered datasets.",
        description="List registered datasets.",
    )
    _add_dataset_list_arguments(list_parser)
    list_parser.set_defaults(handler=run_list_from_args)

    show_parser = dataset_subparsers.add_parser(
        "show",
        help="Show details for a registered dataset.",
        description="Show details for a registered dataset.",
    )
    show_parser.add_argument("name", help="Dataset name from the dataset registry.")
    show_parser.set_defaults(handler=lambda args, p=show_parser: run_show_from_args(args, p))

    open_parser = dataset_subparsers.add_parser(
        "open",
        help="Open a registered dataset folder in file explorer.",
        description="Open a registered dataset folder in file explorer.",
    )
    open_parser.add_argument("name", help="Dataset name from the dataset registry.")
    open_parser.set_defaults(handler=lambda args, p=open_parser: run_open_from_args(args, p))

    readme_parser = dataset_subparsers.add_parser(
        "open-readme",
        help="Open a dataset README, creating the default README if needed.",
        description="Open a dataset README, creating the default README if needed.",
    )
    readme_parser.add_argument("name", help="Dataset name from the dataset registry.")
    readme_parser.set_defaults(
        handler=lambda args, p=readme_parser: run_open_readme_from_args(args, p)
    )

    _add_dataset_rename_parser(dataset_subparsers)
    return parser


def build_parser(*, prog: str = "lisai datasets") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect registered LISAI datasets.", prog=prog)
    subparsers = parser.add_subparsers(dest="datasets_command")
    subparsers.required = True

    list_parser = subparsers.add_parser(
        "list",
        help="List registered datasets.",
        description="List registered datasets.",
    )
    _add_dataset_list_arguments(list_parser)
    list_parser.set_defaults(handler=run_list_from_args)

    show_parser = subparsers.add_parser(
        "show",
        help="Show details for a registered dataset.",
        description="Show details for a registered dataset.",
    )
    show_parser.add_argument("name", help="Dataset name from the dataset registry.")
    show_parser.set_defaults(handler=lambda args, p=show_parser: run_show_from_args(args, p))

    open_parser = subparsers.add_parser(
        "open",
        help="Open a registered dataset folder in file explorer.",
        description="Open a registered dataset folder in file explorer.",
    )
    open_parser.add_argument("name", help="Dataset name from the dataset registry.")
    open_parser.set_defaults(handler=lambda args, p=open_parser: run_open_from_args(args, p))

    readme_parser = subparsers.add_parser(
        "open-readme",
        help="Open a dataset README, creating the default README if needed.",
        description="Open a dataset README, creating the default README if needed.",
    )
    readme_parser.add_argument("name", help="Dataset name from the dataset registry.")
    readme_parser.set_defaults(
        handler=lambda args, p=readme_parser: run_open_readme_from_args(args, p)
    )

    _add_dataset_rename_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = [
    "add_datasets_subparser",
    "build_parser",
    "list_datasets",
    "main",
    "open_dataset",
    "open_dataset_readme",
    "rename_dataset",
    "show_dataset",
]
