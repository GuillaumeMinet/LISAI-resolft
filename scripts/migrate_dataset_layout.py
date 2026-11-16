"""Explicit one-time migration of flat legacy datasets into usage-specific folders.

This is a developer utility for *dataset layout*, distinct from the M2
``lisai workspace migrate`` operation for workspace-global YAMLs.
Nothing is moved on import; use ``--dry-run`` first.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lisai.config.settings import Settings
from lisai.data.dataset_registry import load_dataset_registry
from lisai.infra.paths import Paths
from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.schema import RUN_METADATA_FILENAME


def migrate(*, dry_run: bool = True, paths: Paths) -> int:
    """Return 0 for success, 2 for a preflight conflict; never overwrite data."""
    datasets_root = paths.datasets_root()
    registry = load_dataset_registry(paths.dataset_registry_path())
    moves: list[tuple[Path, Path]] = []
    for name, info in registry.items():
        source = datasets_root / name
        if not source.is_dir():
            continue
        usage = str(info.get("usage", "training"))
        destination = paths.dataset_dir(dataset_name=name, usage=usage)
        if source.resolve() == destination.resolve():
            continue
        moves.append((source, destination))

    collisions = [(old, new) for old, new in moves if new.exists()]
    if collisions:
        for old, new in collisions:
            print(f"Conflict: {new} already exists; will not move {old}")
        return 2

    for source, destination in moves:
        print(f"{source} -> {destination}")
        metadata = sorted(source.rglob(RUN_METADATA_FILENAME))
        rewritten = []
        for old_file in metadata:
            info = read_run_metadata(old_file)
            new_file = destination / old_file.relative_to(source)
            new_relative = new_file.parent.relative_to(paths.datasets_root().parent).as_posix()
            print(f"  run path: {info.path} -> {new_relative}")
            rewritten.append((new_file, info.model_copy(update={"path": new_relative})))
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            for new_file, updated in rewritten:
                write_run_metadata_atomic(new_file, updated)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Already initialized LISAI workspace")
    parser.add_argument("--dry-run", action="store_true", help="Report intended changes without writing")
    parser.add_argument("--apply", action="store_true", help="Explicitly perform the migration")
    args = parser.parse_args(argv)
    if args.dry_run and args.apply:
        parser.error("choose only one of --dry-run and --apply")
    return migrate(dry_run=not args.apply, paths=Paths(Settings.from_workspace(args.workspace)))


if __name__ == "__main__":
    raise SystemExit(main())
