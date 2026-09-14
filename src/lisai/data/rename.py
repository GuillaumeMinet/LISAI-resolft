from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from lisai.config.io.yaml import load_yaml
from lisai.data.data_loaders.split_manifest import read_split_manifest
from lisai.data.dataset_registry import load_dataset_registry
from lisai.infra.paths import Paths
from lisai.infra.paths.run_location import RUN_ARCHIVE_DIRNAME
from lisai.runs.schema import RUN_METADATA_FILENAME, RUN_NON_TERMINAL_STATUSES, RunMetadata


class DatasetRenameError(ValueError):
    """Raised when a dataset rename cannot be planned or completed safely."""


@dataclass(frozen=True)
class PlannedFileRewrite:
    relative_path: Path
    kind: str
    original_bytes: bytes
    updated_bytes: bytes


@dataclass(frozen=True)
class DatasetRenamePlan:
    old_name: str
    new_name: str
    usage: str
    source_dir: Path
    destination_dir: Path
    registry_path: Path
    registry_original_bytes: bytes
    registry_updated_bytes: bytes
    rewrites: tuple[PlannedFileRewrite, ...]
    active_run_metadata_count: int = 0
    training_config_count: int = 0
    split_manifest_count: int = 0
    archived_run_metadata_count: int = 0


def _validate_dataset_component(name: str, *, label: str) -> str:
    text = str(name).strip()
    if not text:
        raise DatasetRenameError(f"{label} must not be empty.")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise DatasetRenameError(
            f"{label} must be a single dataset folder name, got {name!r}."
        )
    if Path(text).is_absolute():
        raise DatasetRenameError(f"{label} must not be an absolute path.")
    return text


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    text = yaml.safe_dump(dict(payload), sort_keys=False)
    return text.encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def _rewrite_stored_dataset_path(
    stored_path: str,
    *,
    old_prefix: str,
    new_prefix: str,
) -> str:
    normalized = stored_path.replace("\\", "/").strip("/")
    old_prefix = old_prefix.strip("/")
    new_prefix = new_prefix.strip("/")
    if normalized == old_prefix:
        return new_prefix
    prefix = f"{old_prefix}/"
    if normalized.startswith(prefix):
        return f"{new_prefix}/{normalized[len(prefix):]}"
    raise DatasetRenameError(
        "Run metadata path does not point inside the dataset being renamed: "
        f"{stored_path!r}. Expected prefix {old_prefix!r}."
    )


def _run_rewrites(
    *,
    paths: Paths,
    source_dir: Path,
    old_name: str,
    new_name: str,
) -> tuple[tuple[PlannedFileRewrite, ...], int, int, int, int]:
    runs_dir = paths.dataset_runs_dir_from_dataset_dir(source_dir)
    if not runs_dir.is_dir():
        return (), 0, 0, 0, 0

    stored_root = paths.datasets_root().parent.resolve()
    old_prefix = source_dir.resolve().relative_to(stored_root).as_posix()
    new_dataset_dir = source_dir.with_name(new_name).resolve()
    new_prefix = new_dataset_dir.relative_to(stored_root).as_posix()

    rewrites: list[PlannedFileRewrite] = []
    active_metadata_count = 0
    config_count = 0
    split_count = 0
    archived_metadata_count = 0

    metadata_paths = sorted(runs_dir.rglob(RUN_METADATA_FILENAME))
    for metadata_path in metadata_paths:
        relative_to_runs = metadata_path.relative_to(runs_dir)
        archived = RUN_ARCHIVE_DIRNAME in relative_to_runs.parts
        relative_to_dataset = metadata_path.relative_to(source_dir)

        original = metadata_path.read_bytes()
        try:
            payload = json.loads(original.decode("utf-8"))
            metadata = RunMetadata.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
            raise DatasetRenameError(
                f"Invalid run metadata prevents safe dataset rename: {metadata_path}: {exc}"
            ) from exc

        if metadata.dataset != old_name:
            raise DatasetRenameError(
                "Run metadata dataset does not match the dataset being renamed: "
                f"{metadata_path} has {metadata.dataset!r}, expected {old_name!r}."
            )

        if not archived and metadata.status in RUN_NON_TERMINAL_STATUSES:
            raise DatasetRenameError(
                "Dataset has a non-terminal training run and cannot be renamed safely: "
                f"{metadata_path.parent} (status={metadata.status!r})."
            )

        updated = metadata.model_copy(
            update={
                "dataset": new_name,
                "path": _rewrite_stored_dataset_path(
                    metadata.path,
                    old_prefix=old_prefix,
                    new_prefix=new_prefix,
                ),
            }
        )
        rewrites.append(
            PlannedFileRewrite(
                relative_path=relative_to_dataset,
                kind="archived_run_metadata" if archived else "active_run_metadata",
                original_bytes=original,
                updated_bytes=_json_bytes(updated.model_dump(mode="json")),
            )
        )

        if archived:
            archived_metadata_count += 1
            continue

        active_metadata_count += 1
        run_dir = metadata_path.parent

        config_path = paths.cfg_train_path(run_dir=run_dir)
        if config_path.is_file():
            original_config = config_path.read_bytes()
            try:
                config = load_yaml(config_path)
            except Exception as exc:
                raise DatasetRenameError(
                    f"Could not read saved training config {config_path}: {exc}"
                ) from exc
            data = config.get("data")
            if not isinstance(data, Mapping):
                raise DatasetRenameError(
                    f"Saved training config has no valid `data` section: {config_path}"
                )
            current_dataset = data.get("dataset_name")
            if current_dataset != old_name:
                raise DatasetRenameError(
                    "Saved training config dataset does not match the dataset being renamed: "
                    f"{config_path} has {current_dataset!r}, expected {old_name!r}."
                )
            updated_config = dict(config)
            updated_data = dict(data)
            updated_data["dataset_name"] = new_name
            updated_config["data"] = updated_data
            rewrites.append(
                PlannedFileRewrite(
                    relative_path=config_path.relative_to(source_dir),
                    kind="training_config",
                    original_bytes=original_config,
                    updated_bytes=_yaml_bytes(updated_config),
                )
            )
            config_count += 1

        manifest_path = paths.split_manifest_path(run_dir=run_dir)
        if manifest_path.is_file():
            original_manifest = manifest_path.read_bytes()
            try:
                manifest = read_split_manifest(manifest_path)
            except Exception as exc:
                raise DatasetRenameError(
                    f"Could not read split manifest {manifest_path}: {exc}"
                ) from exc
            current_dataset = manifest.get("dataset_name")
            if current_dataset != old_name:
                raise DatasetRenameError(
                    "Split manifest dataset does not match the dataset being renamed: "
                    f"{manifest_path} has {current_dataset!r}, expected {old_name!r}."
                )
            updated_manifest = dict(manifest)
            updated_manifest["dataset_name"] = new_name
            rewrites.append(
                PlannedFileRewrite(
                    relative_path=manifest_path.relative_to(source_dir),
                    kind="split_manifest",
                    original_bytes=original_manifest,
                    updated_bytes=_json_bytes(updated_manifest),
                )
            )
            split_count += 1

    return (
        tuple(rewrites),
        active_metadata_count,
        config_count,
        split_count,
        archived_metadata_count,
    )


def build_dataset_rename_plan(
    old_name: str,
    new_name: str,
    *,
    paths: Paths,
) -> DatasetRenamePlan:
    old_name = _validate_dataset_component(old_name, label="Old dataset name")
    new_name = _validate_dataset_component(new_name, label="New dataset name")
    if old_name == new_name:
        raise DatasetRenameError("Old and new dataset names are identical.")

    registry_path = paths.dataset_registry_path()
    registry = load_dataset_registry(registry_path)
    info = registry.get(old_name)
    if info is None:
        known = ", ".join(sorted(registry)) or "none"
        raise DatasetRenameError(
            f"Unknown dataset {old_name!r}. Known datasets: {known}."
        )
    if new_name in registry:
        raise DatasetRenameError(f"Dataset {new_name!r} already exists in the registry.")

    usage = str(info.get("usage") or "training").strip().lower()
    source_dir = paths.dataset_dir(dataset_name=old_name, usage=usage)
    destination_dir = paths.dataset_dir(dataset_name=new_name, usage=usage)
    if not source_dir.is_dir():
        raise DatasetRenameError(f"Dataset folder does not exist: {source_dir}")
    if destination_dir.exists():
        raise DatasetRenameError(f"Destination dataset folder already exists: {destination_dir}")

    casefold_matches = [name for name in registry if name.casefold() == new_name.casefold()]
    if casefold_matches:
        raise DatasetRenameError(
            f"Dataset name {new_name!r} conflicts case-insensitively with "
            f"{casefold_matches[0]!r}."
        )

    rewrites: tuple[PlannedFileRewrite, ...] = ()
    active_metadata_count = 0
    config_count = 0
    split_count = 0
    archived_metadata_count = 0
    if usage == "training":
        (
            rewrites,
            active_metadata_count,
            config_count,
            split_count,
            archived_metadata_count,
        ) = _run_rewrites(
            paths=paths,
            source_dir=source_dir,
            old_name=old_name,
            new_name=new_name,
        )

    registry_original_bytes = registry_path.read_bytes()
    updated_registry: dict[str, dict[str, Any]] = {}
    for name, entry in registry.items():
        updated_registry[new_name if name == old_name else name] = dict(entry)

    return DatasetRenamePlan(
        old_name=old_name,
        new_name=new_name,
        usage=usage,
        source_dir=source_dir,
        destination_dir=destination_dir,
        registry_path=registry_path,
        registry_original_bytes=registry_original_bytes,
        registry_updated_bytes=_yaml_bytes(updated_registry),
        rewrites=rewrites,
        active_run_metadata_count=active_metadata_count,
        training_config_count=config_count,
        split_manifest_count=split_count,
        archived_run_metadata_count=archived_metadata_count,
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def apply_dataset_rename(plan: DatasetRenamePlan) -> None:
    if not plan.source_dir.is_dir():
        raise DatasetRenameError(f"Dataset folder disappeared before rename: {plan.source_dir}")
    if plan.destination_dir.exists():
        raise DatasetRenameError(
            f"Destination appeared before rename could start: {plan.destination_dir}"
        )

    renamed = False
    registry_changed = False
    try:
        plan.source_dir.rename(plan.destination_dir)
        renamed = True

        for rewrite in plan.rewrites:
            _atomic_write_bytes(plan.destination_dir / rewrite.relative_path, rewrite.updated_bytes)

        _atomic_write_bytes(plan.registry_path, plan.registry_updated_bytes)
        registry_changed = True
    except Exception as exc:
        rollback_errors: list[str] = []
        if renamed:
            for rewrite in plan.rewrites:
                try:
                    _atomic_write_bytes(
                        plan.destination_dir / rewrite.relative_path,
                        rewrite.original_bytes,
                    )
                except Exception as rollback_exc:  # pragma: no cover - exceptional recovery path
                    rollback_errors.append(
                        f"restore {rewrite.relative_path}: {rollback_exc}"
                    )
        if registry_changed or plan.registry_path.exists():
            try:
                _atomic_write_bytes(plan.registry_path, plan.registry_original_bytes)
            except Exception as rollback_exc:  # pragma: no cover - exceptional recovery path
                rollback_errors.append(f"restore registry: {rollback_exc}")
        if renamed and plan.destination_dir.exists() and not plan.source_dir.exists():
            try:
                plan.destination_dir.rename(plan.source_dir)
            except Exception as rollback_exc:  # pragma: no cover - exceptional recovery path
                rollback_errors.append(f"restore dataset folder: {rollback_exc}")

        message = f"Dataset rename failed and rollback was attempted: {exc}"
        if rollback_errors:
            message += "\nRollback issues:\n  - " + "\n  - ".join(rollback_errors)
        raise DatasetRenameError(message) from exc


__all__ = [
    "DatasetRenameError",
    "DatasetRenamePlan",
    "PlannedFileRewrite",
    "apply_dataset_rename",
    "build_dataset_rename_plan",
]
