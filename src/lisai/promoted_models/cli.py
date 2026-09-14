from __future__ import annotations

import argparse
from typing import Sequence, get_args

from lisai.config.models.training import TaskName

from .install import install_model_archive
from .package import (
    export_promoted_model,
    load_promoted_model, 
    set_promoted_model_task,
    sync_promoted_model
)
from .remove import remove_promoted_model
from .registry import load_promoted_model_registry

VALID_TASK_NAMES  = ", ".join(get_args(TaskName))

def _render_models_table() -> str:
    registry = load_promoted_model_registry()
    if not registry.models:
        return "No promoted models found."

    rows: list[tuple[str, str, str, str, str]] = []
    for name in sorted(registry.models):
        entry = registry.models[name]
        task = entry.task or "-"
        rows.append((name, task, entry.origin, entry.source_run_id, entry.path))
    headers = ("name", "task", "origin", "source_run_id", "path")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) 
              for i in range(len(headers))]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return "\n".join(lines)


def run_list_from_args(args: argparse.Namespace) -> int:
    print(_render_models_table())
    return 0


def run_set_task_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try: 
        promoted = set_promoted_model_task(
            args.name, args.task
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Updated model: {promoted.manifest.name}")
    print(f"Task: {promoted.manifest.model.task}")
    return 0

def run_sync_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        promoted = sync_promoted_model(args.name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Synchronized model: {promoted.manifest.name}")
    print(f"Task: {promoted.manifest.model.task}")
    return 0

def run_show_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        promoted = load_promoted_model(args.name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    manifest = promoted.manifest
    registry = load_promoted_model_registry()
    entry = registry.models.get(manifest.name)
    print(f"Name: {manifest.name}")
    print(f"Path: {promoted.model_dir}")
    if entry is not None:
        print(f"Origin: {entry.origin}")
    print(f"Task: {manifest.model.task}")
    print(f"Architecture: {manifest.model.architecture}")
    print(f"Dataset: {manifest.training_data.dataset}")
    print(f"Source run: {manifest.source.run_name} ({manifest.source.run_id})")
    print(f"Source status: {manifest.source.run_status}")
    print(f"Checkpoint: {manifest.source.checkpoint_selector}")
    if manifest.source.code is not None:
        print(f"Git commit: {manifest.source.code.git_commit or '-'}")
        print(f"LISAI version: {manifest.source.code.lisai_version or '-'}")
    return 0


def run_export_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        exported = export_promoted_model(
            args.name,
            output=args.output,
            overwrite=args.overwrite,
        )
    except (KeyError, FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    print(f"Exported model: {exported.manifest.name}")
    print(f"Archive: {exported.archive_path}")
    print(f"SHA256: {exported.archive_sha256}")
    return 0


def run_install_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        installed = install_model_archive(
            args.archive,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    print(f"Installed model: {installed.model.manifest.name}")
    print(f"Path: {installed.model.model_dir}")
    return 0


def run_remove_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    registry = load_promoted_model_registry()
    entry = registry.models.get(args.name)
    if entry is None:
        parser.exit(status=1, message=f"Unknown promoted model {args.name!r}.\n")

    if not args.yes:
        answer = input(
            f"Remove promoted model {args.name!r} from the local library? "
            "Exported archives will be kept. [y/N]: "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Removal cancelled.")
            return 0

    try:
        removed = remove_promoted_model(args.name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    print(f"Removed model: {removed.name}")
    print(f"Removed path: {removed.model_dir}")
    return 0


def _add_model_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="models_command")
    subparsers.required = True

    list_parser = subparsers.add_parser(
        "list",
        help="List locally promoted reusable models.",
        description="List locally promoted reusable models.",
    )
    list_parser.set_defaults(handler=run_list_from_args)

    show_parser = subparsers.add_parser(
        "show",
        help="Show one locally promoted model.",
        description="Show metadata for one locally promoted model.",
    )
    show_parser.add_argument("name", help="Public promoted-model name.")
    show_parser.set_defaults(handler=lambda args, p=show_parser: run_show_from_args(args, p))

    set_task_parser = subparsers.add_parser(
        "set-task",
        help="Set the task metadata of a promoted model.",
        description=(
            "Update the canonical task metadata of a promoted model and "
            "synchronize its registry entry and model card. Task should be "
            f"a valid LISAI TaskName."
        ),
    )
    set_task_parser.add_argument(
        "name",
        help="Public promoted-model name.",
    )
    set_task_parser.add_argument(
        "task",
        help=f"New promoted-model task. One of: {VALID_TASK_NAMES}",
    )
    set_task_parser.set_defaults(
        handler=lambda args, p=set_task_parser: run_set_task_from_args(args, p)
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help="Synchronize derived metadata for a promoted model.",
        description=(
            "Synchronize the local registry and model card from "
            "lisai_model.yaml."
        ),
    )
    sync_parser.add_argument(
        "name",
        help="Public promoted-model name.",
    )
    sync_parser.set_defaults(
        handler=lambda args, p=sync_parser: run_sync_from_args(args, p)
    )


    install_parser = subparsers.add_parser(
        "install",
        help="Install a local .lisai.zip archive into the promoted-model library.",
        description="Validate and install a portable LISAI promoted-model archive.",
    )
    install_parser.add_argument("archive", help="Path to a local .lisai.zip archive.")
    install_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a locally registered model with the same public name.",
    )
    install_parser.set_defaults(
        handler=lambda args, p=install_parser: run_install_from_args(args, p)
    )

    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove a model from the local promoted-model library.",
        description=(
            "Remove a locally promoted or installed model. Exported .lisai.zip archives "
            "are left untouched."
        ),
    )
    remove_parser.add_argument("name", help="Public promoted-model name.")
    remove_parser.add_argument(
        "--yes",
        action="store_true",
        help="Remove without asking for confirmation.",
    )
    remove_parser.set_defaults(
        handler=lambda args, p=remove_parser: run_remove_from_args(args, p)
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export a locally promoted model as a portable .lisai.zip archive.",
        description="Export a locally promoted model for publication or transfer.",
    )
    export_parser.add_argument("name", help="Public promoted-model name.")
    export_parser.add_argument(
        "--output",
        help=(
            "Output ZIP path or directory. Defaults to "
            "<data_root>/models/exports/<name>.lisai.zip."
        ),
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing export archive at the output path.",
    )
    export_parser.set_defaults(handler=lambda args, p=export_parser: run_export_from_args(args, p))
    return parser


def add_models_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "models",
        help="Manage locally promoted LISAI models.",
        description="List, inspect, install, remove, and export locally promoted LISAI models.",
    )
    return _add_model_commands(parser)


def build_parser(*, prog: str = "lisai models") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="List, inspect, install, remove, and export locally promoted LISAI models.",
    )
    return _add_model_commands(parser)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = ["add_models_subparser", "build_parser", "main"]
