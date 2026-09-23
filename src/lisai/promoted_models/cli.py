from __future__ import annotations

import argparse
from typing import Sequence, get_args

from lisai.config.models.training import TaskName
from lisai.infra.cli.prompts import prompt_yes_no
from lisai.infra.cli.selection import resolve_partial_name

from . import catalog
from .download import (
    DownloadConflictError,
    DownloadIntegrityError,
    ModelDownloadError,
    download_model,
)
from .install import install_model_archive
from .package import (
    export_promoted_model,
    load_promoted_model, 
    set_promoted_model_config,
    set_promoted_model_task,
    sync_promoted_model
)
from .remove import remove_promoted_model
from .registry import load_promoted_model_registry
from .sources.zenodo import ZenodoSourceError

VALID_TASK_NAMES  = ", ".join(get_args(TaskName))
_MODEL_LIST_HINT = "Use 'lisai models list' to inspect available promoted models."


def _resolve_model_name(
    name: str,
    *,
    parser: argparse.ArgumentParser,
) -> str:
    registry = load_promoted_model_registry()
    resolved = resolve_partial_name(
        name,
        registry.models,
        entity_name="promoted model",
        column_name="model",
        help_hint=_MODEL_LIST_HINT,
    )
    if resolved is None:
        parser.exit(status=1)
    return resolved

def _render_models_table() -> str:
    registry = load_promoted_model_registry()
    if not registry.models:
        return "No promoted models found."

    rows: list[tuple[str, str, str, str, str]] = []
    for name in sorted(registry.models):
        entry = registry.models[name]
        task = entry.task or "-"
        rows.append((name, task, entry.origin, entry.source_run_id))
    headers = ("name", "task", "origin", "source_run_id")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) 
              for i in range(len(headers))]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return "\n".join(lines)


def run_list_from_args(args: argparse.Namespace) -> int:
    print(_render_models_table())
    return 0


def _render_catalog_table() -> str:
    models = catalog.list_models()
    if not models:
        return "No downloadable models found."

    rows = [(model.name, model.task, model.description) for model in models]
    headers = ("name", "task", "description")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return "\n".join(lines)


def run_catalog_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        table = _render_catalog_table()
    except (catalog.CatalogUnavailableError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    print(table)
    return 0


def _print_download_summary(result) -> None:
    if result.status == "reused":
        print(f"Model already downloaded: {result.name}")
        print("Existing archive checksum verified; reusing it.")
    elif result.status == "overwritten":
        print(f"Downloaded model: {result.name}")
        print("Replaced the existing archive after checksum mismatch.")
    else:
        print(f"Downloaded model: {result.name}")
    print(f"Archive: {result.archive_path}")
    print("SHA256: verified")


def _download_one_from_catalog(
    name: str,
    *,
    overwrite: bool,
    install: bool,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        result = download_model(name, overwrite=overwrite)
    except DownloadConflictError as exc:
        confirmed = prompt_yes_no(
            "An existing downloaded archive does not match the catalog checksum. "
            f"Replace it?\n  {exc.path}\n[y/N]: ",
            input_fn=input,
        )
        if not confirmed:
            print(f"Skipped model: {name}")
            return 0
        try:
            result = download_model(name, overwrite=True)
        except (
            catalog.CatalogUnavailableError,
            DownloadIntegrityError,
            ModelDownloadError,
            ZenodoSourceError,
            KeyError,
            ValueError,
        ) as retry_exc:
            parser.exit(status=1, message=f"{retry_exc}\n")
    except (
        catalog.CatalogUnavailableError,
        DownloadIntegrityError,
        ModelDownloadError,
        ZenodoSourceError,
        KeyError,
        ValueError,
    ) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    _print_download_summary(result)

    if install:
        try:
            installed = install_model_archive(result.archive_path)
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            parser.exit(status=1, message=f"{exc}\n")
        print()
        print(f"Installed model: {installed.model.manifest.name}")
        print(f"Path: {installed.model.model_dir}")
    else:
        print()
        print("To install:")
        print(f"  lisai models install {result.archive_path.name}")
    return 0


def run_download_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.all and args.name is not None:
        parser.error("A model name cannot be combined with --all.")
    if not args.all and args.name is None:
        parser.error("Provide a model name, or use --all.")

    if not args.all:
        return _download_one_from_catalog(
            args.name,
            overwrite=args.overwrite,
            install=args.install,
            parser=parser,
        )

    try:
        models = catalog.list_models()
    except (catalog.CatalogUnavailableError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    if not models:
        print("No downloadable models found.")
        return 0

    installed_names = set(load_promoted_model_registry().models) if args.install else set()
    for index, model in enumerate(models):
        if index:
            print("\n" + "-" * 60 + "\n")
        if args.install and model.name in installed_names:
            print(f"Model already installed: {model.name}")
            continue
        _download_one_from_catalog(
            model.name,
            overwrite=args.overwrite,
            install=args.install,
            parser=parser,
        )
    return 0


def run_set_task_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    name = _resolve_model_name(args.name, parser=parser)
    try: 
        promoted = set_promoted_model_task(
            name, args.task
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Updated model: {promoted.manifest.name}")
    print(f"Task: {promoted.manifest.model.task}")
    return 0

def run_set_config_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if args.clear and args.config is not None:
        parser.error("CONFIG cannot be provided together with --clear.")
    if not args.clear and args.config is None:
        parser.error("Provide CONFIG, or use --clear to remove the model inference config.")

    name = _resolve_model_name(args.name, parser=parser)
    try:
        promoted = set_promoted_model_config(
            name,
            None if args.clear else args.config,
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Updated model: {promoted.manifest.name}")
    print(f"Inference config: {promoted.manifest.artifacts.inference_config or 'none'}")
    return 0


def run_sync_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    name = _resolve_model_name(args.name, parser=parser)
    try:
        promoted = sync_promoted_model(name)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Synchronized model: {promoted.manifest.name}")
    print(f"Task: {promoted.manifest.model.task}")
    return 0

def run_show_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    name = _resolve_model_name(args.name, parser=parser)
    try:
        promoted = load_promoted_model(name)
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
    print(f"Inference config: {manifest.artifacts.inference_config or '-'}")
    return 0


def run_export_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    name = _resolve_model_name(args.name, parser=parser)
    try:
        exported = export_promoted_model(
            name,
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
    name = _resolve_model_name(args.name, parser=parser)

    if not args.yes:
        confirmed = prompt_yes_no(
            f"Remove promoted model {name!r} from the local library? "
            "Exported archives will be kept. [y/N]: ",
            input_fn=input,
        )
        if not confirmed:
            print("Removal cancelled.")
            return 0

    try:
        removed = remove_promoted_model(name)
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

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="List models available from the remote LISAI download catalog.",
        description="List models available from the remote LISAI download catalog.",
    )
    catalog_parser.set_defaults(
        handler=lambda args, p=catalog_parser: run_catalog_from_args(args, p)
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download a model from the LISAI model catalog.",
        description=(
            "Download a named model from the LISAI model catalog into the configured "
            "promoted-model downloads directory."
        ),
    )
    download_parser.add_argument(
        "name",
        nargs="?",
        help="Exact model name from 'lisai models catalog'.",
    )
    download_parser.add_argument(
        "--all",
        action="store_true",
        help="Download every model in the remote catalog.",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing downloaded archive when its checksum does not match.",
    )
    download_parser.add_argument(
        "--install",
        action="store_true",
        help="Install the verified archive immediately after downloading it.",
    )
    download_parser.set_defaults(
        handler=lambda args, p=download_parser: run_download_from_args(args, p)
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show one locally promoted model.",
        description="Show metadata for one locally promoted model.",
    )
    show_parser.add_argument("name", help="Public promoted-model name or partial name.")
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
        help="Public promoted-model name or partial name.",
    )
    set_task_parser.add_argument(
        "task",
        help=f"New promoted-model task. One of: {VALID_TASK_NAMES}",
    )
    set_task_parser.set_defaults(
        handler=lambda args, p=set_task_parser: run_set_task_from_args(args, p)
    )

    set_config_parser = subparsers.add_parser(
        "set-config",
        help="Set the default inference config of a promoted model.",
        description=(
            "Attach, replace, or clear the sparse default apply config stored with a promoted "
            "model. The stored config is included automatically in later model exports."
        ),
    )
    set_config_parser.add_argument(
        "name",
        help="Public promoted-model name or partial name.",
    )
    set_config_parser.add_argument(
        "config",
        nargs="?",
        help=(
            "Inference config path or name from configs/inference. The source config is copied "
            "into the promoted model as config_inference.yaml."
        ),
    )
    set_config_parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the promoted model's default inference config.",
    )
    set_config_parser.set_defaults(
        handler=lambda args, p=set_config_parser: run_set_config_from_args(args, p)
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
        help="Public promoted-model name or partial name.",
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
    remove_parser.add_argument("name", help="Public promoted-model name or partial name.")
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
    export_parser.add_argument("name", help="Public promoted-model name or partial name.")
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
        help="Manage promoted and downloadable LISAI models.",
        description=(
            "List, inspect, download, install, remove, and export LISAI promoted models."
        ),
    )
    return _add_model_commands(parser)


def build_parser(*, prog: str = "lisai models") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="List, inspect, download, install, remove, and export LISAI promoted models.",
    )
    return _add_model_commands(parser)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = ["add_models_subparser", "build_parser", "main"]
