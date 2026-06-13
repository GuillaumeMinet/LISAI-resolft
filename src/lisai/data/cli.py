from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.runs.cli import _try_open_path

from .dataset_registry import load_dataset_registry


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


def list_datasets(*, paths: Paths | None = None) -> None:
    paths = paths or _paths()
    registry = _registry_from_paths(paths)
    if not registry:
        print(f"No datasets found in {paths.dataset_registry_path()}")
        return

    headers = ("name", "usage", "format", "types", "files", "frames", "split", "range", "defaults")
    rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
    for name, info in sorted(registry.items()):
        data_types = _data_types(info)
        rows.append(
            (
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
        )
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
    return info, paths.dataset_dir(dataset_name=name)


def show_dataset(name: str, *, paths: Paths | None = None, parser: argparse.ArgumentParser) -> None:
    paths = paths or _paths()
    info, dataset_dir = _require_dataset(name, paths=paths, parser=parser)

    print(f"Dataset: {name}")
    print(f"Path: {dataset_dir.resolve()}")
    print(f"Usage: {info.get('usage') or '-'}")
    print(f"Format: {info.get('data_format') or '-'}")

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


def open_dataset(name: str, *, paths: Paths | None = None, parser: argparse.ArgumentParser) -> int:
    paths = paths or _paths()
    _, dataset_dir = _require_dataset(name, paths=paths, parser=parser)
    resolved = dataset_dir.resolve()
    if _try_open_path(resolved):
        return 0
    print(resolved)
    return 0


def run_list_from_args(args: argparse.Namespace) -> int:
    list_datasets()
    return 0


def run_show_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    show_dataset(args.name, parser=parser)
    return 0


def run_open_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    return open_dataset(args.name, parser=parser)


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
    "show_dataset",
]
