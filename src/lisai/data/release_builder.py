"""Build and validate a local dataset release catalog.

The release builder is intentionally local-only: the LISAI data root provides
registry metadata, while the staging directory is authoritative for the curated
release contents. It does not contact Zenodo and it does not create archives.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, TextIO

import yaml
from lisai.data.dataset_registry import load_dataset_registry

DATASET_CATALOG_SCHEMA_VERSION = 1
_DATASET_REGISTRY_RELATIVE_PATH = Path("datasets") / "dataset_registry.yml"
_IGNORED_ZIP_TOP_LEVELS = {"application"}
_NOISE_MODELS_TOP_LEVEL = "noise_models"
_GIB = 1024**3


class ReleaseValidationError(ValueError):
    """Raised when the staged release is inconsistent with the LISAI data root."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("Release validation failed:\n" + "\n".join(f"- {e}" for e in self.errors))


@dataclass(frozen=True)
class ArchiveInfo:
    path: Path
    top_level: str
    files: dict[str, int]


@dataclass(frozen=True)
class DatasetDependencies:
    evaluation_datasets: tuple[str, ...]
    noise_models: tuple[str, ...]


def _format_bytes(size: int) -> str:
    if size >= _GIB:
        return f"{size / _GIB:.2f} GiB"
    mib = 1024**2
    if size >= mib:
        return f"{size / mib:.1f} MiB"
    kib = 1024
    if size >= kib:
        return f"{size / kib:.1f} KiB"
    return f"{size} B"


def _normalized_zip_name(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def _inspect_zip(path: Path) -> ArchiveInfo:
    """Read ZIP metadata without extracting file contents."""
    top_levels: set[str] = set()
    files: dict[str, int] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                normalized = _normalized_zip_name(info.filename)
                if not normalized:
                    continue
                parts = PurePosixPath(normalized).parts
                if not parts or parts[0] == "__MACOSX":
                    continue
                top_levels.add(parts[0])
                if info.is_dir():
                    continue
                if normalized in files:
                    raise ReleaseValidationError(
                        [f"Archive {path.name!r} contains duplicate member {normalized!r}."]
                    )
                files[normalized] = int(info.file_size)
    except zipfile.BadZipFile as exc:
        raise ReleaseValidationError(
            [f"Archive {path.name!r} is not a valid ZIP file: {exc}"]
        ) from exc

    if len(top_levels) != 1:
        rendered = ", ".join(sorted(top_levels)) or "<none>"
        raise ReleaseValidationError(
            [
                f"Archive {path.name!r} must contain exactly one top-level directory; "
                f"found: {rendered}."
            ]
        )

    return ArchiveInfo(path=path, top_level=next(iter(top_levels)), files=files)


def _relative_archive_files(archive: ArchiveInfo) -> dict[str, int]:
    prefix = f"{archive.top_level}/"
    relative: dict[str, int] = {}
    for member, size in archive.files.items():
        if not member.startswith(prefix):
            continue
        rel = member[len(prefix) :]
        if rel:
            relative[rel] = size
    return relative


def _validate_dataset_archives(
    *,
    dataset_name: str,
    archives: list[ArchiveInfo],
) -> list[str]:
    """Validate the staged representation of one logical dataset.

    The staged release is intentionally allowed to be a curated subset (or
    superset) of the local source dataset.  We therefore only validate the
    staged archives themselves here: files may be omitted or added relative to
    the source root, but the same relative file must not be supplied twice
    across multiple archives for the same dataset.
    """
    errors: list[str] = []
    archived: set[str] = set()
    duplicate_members: set[str] = set()

    for archive in archives:
        relative_files = set(_relative_archive_files(archive))
        duplicate_members.update(archived & relative_files)
        archived.update(relative_files)

    if duplicate_members:
        preview = ", ".join(sorted(duplicate_members)[:5])
        suffix = " ..." if len(duplicate_members) > 5 else ""
        errors.append(
            f"Dataset {dataset_name!r} has files duplicated across its archives: "
            f"{preview}{suffix}"
        )

    if not archived:
        errors.append(f"Dataset {dataset_name!r} has no files in its staged archive(s).")

    return errors


def _load_archived_yaml(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    archive_name: str,
) -> dict[str, Any]:
    location = f"{archive_name}:{_normalized_zip_name(info.filename)}"
    try:
        raw = archive.read(info)
        payload = yaml.safe_load(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, yaml.YAMLError, OSError) as exc:
        raise ReleaseValidationError([f"Could not read YAML metadata {location}: {exc}"]) from exc

    if not isinstance(payload, Mapping):
        raise ReleaseValidationError(
            [f"YAML metadata {location} must contain a mapping at the document root."]
        )
    return dict(payload)


def _noise_model_name(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        name = value.get("name")
        if name is not None:
            text = str(name).strip()
            return text or None
    return None


def _discover_dependencies(archives: list[ArchiveInfo]) -> DatasetDependencies:
    """Discover dependencies from metadata that is actually present in the release.

    This deliberately scans the staged ZIP contents rather than the local source
    dataset.  A release may omit old runs/evaluations or add curated files, and
    only metadata shipped to the user should affect its catalog dependencies.
    """
    evaluation_datasets: set[str] = set()
    noise_models: set[str] = set()

    for archive_info in archives:
        prefix = f"{archive_info.top_level}/"
        try:
            with zipfile.ZipFile(archive_info.path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    normalized = _normalized_zip_name(info.filename)
                    if not normalized.startswith(prefix):
                        continue
                    relative = normalized[len(prefix) :]
                    parts = PurePosixPath(relative).parts
                    if "runs" not in parts:
                        continue

                    if parts[-1] == "config_train.yaml":
                        cfg = _load_archived_yaml(
                            archive,
                            info,
                            archive_name=archive_info.path.name,
                        )
                        noise_model = _noise_model_name(cfg.get("noise_model"))
                        if noise_model:
                            noise_models.add(noise_model)
                        continue

                    if parts[-1] != "evaluation.yaml":
                        continue

                    metadata = _load_archived_yaml(
                        archive,
                        info,
                        archive_name=archive_info.path.name,
                    )
                    dataset = metadata.get("dataset")
                    if not isinstance(dataset, Mapping):
                        continue
                    usage = str(dataset.get("usage") or "").strip().lower()
                    name = dataset.get("name")
                    if usage == "evaluation" and name is not None:
                        text = str(name).strip()
                        if text:
                            evaluation_datasets.add(text)
        except zipfile.BadZipFile as exc:
            # _inspect_zip normally catches this first, but keep dependency
            # discovery independently safe if called directly.
            raise ReleaseValidationError(
                [f"Archive {archive_info.path.name!r} is not a valid ZIP file: {exc}"]
            ) from exc

    return DatasetDependencies(
        evaluation_datasets=tuple(sorted(evaluation_datasets, key=str.casefold)),
        noise_models=tuple(sorted(noise_models, key=str.casefold)),
    )


def _sha256_file(path: Path, *, progress: bool, stream: TextIO) -> str:
    total = path.stat().st_size
    digest = hashlib.sha256()
    read_total = 0
    last_percent = -1

    if progress:
        print(f"Hashing {path.name} ({_format_bytes(total)})", file=stream)

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            read_total += len(chunk)
            if progress and total:
                percent = int(read_total * 100 / total)
                if percent >= last_percent + 10 or percent == 100:
                    last_percent = percent
                    print(f"  {percent:3d}%", file=stream)

    return digest.hexdigest()


def _archive_record(path: Path, *, progress: bool, stream: TextIO) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path, progress=progress, stream=stream),
    }


def _noise_model_dirs_in_archive(archive: ArchiveInfo) -> set[str]:
    prefix = f"{archive.top_level}/"
    names: set[str] = set()
    for member in archive.files:
        if not member.startswith(prefix):
            continue
        remainder = member[len(prefix) :]
        parts = PurePosixPath(remainder).parts
        if parts:
            names.add(parts[0])
    return names


def _validate_noise_models(
    *,
    archive: ArchiveInfo | None,
    required: set[str],
) -> list[str]:
    """Ensure every noise model required by staged runs is shipped and usable."""
    errors: list[str] = []
    if not required:
        return errors
    if archive is None:
        return [
            "At least one staged dataset references a noise model, but no ZIP with top-level "
            f"directory {_NOISE_MODELS_TOP_LEVEL!r} is present in the staging directory."
        ]

    archived_models = _noise_model_dirs_in_archive(archive)
    for name in sorted(required, key=str.casefold):
        if name not in archived_models:
            errors.append(
                f"Noise model {name!r} is referenced by a staged dataset but is absent from "
                f"{archive.path.name!r}."
            )
            continue

        for required_file in ("GMM.npz", "norm_prm.json"):
            archive_member = f"{archive.top_level}/{name}/{required_file}"
            if archive_member not in archive.files:
                errors.append(
                    f"Noise model {name!r} is required but {required_file!r} is missing from "
                    f"{archive.path.name!r}."
                )
    return errors


def build_dataset_catalog(
    data_root: str | Path,
    staging_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    progress: bool = True,
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Validate staged release archives and write ``dataset_catalog.json``.

    The staging directory is authoritative for release membership. Dataset ZIPs
    are identified by their single top-level directory, which must match a
    registered dataset name. ``application`` archives and ``*.lisai.zip`` model
    packages are intentionally outside the dataset catalog.
    """
    stream = stream or sys.stdout
    data_root = Path(data_root).expanduser().resolve()
    staging_dir = Path(staging_dir).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else staging_dir / "dataset_catalog.json"
    )

    errors: list[str] = []
    warnings: list[str] = []

    if not data_root.is_dir():
        raise ReleaseValidationError([f"LISAI data root does not exist: {data_root}"])
    if not staging_dir.is_dir():
        raise ReleaseValidationError([f"Staging directory does not exist: {staging_dir}"])

    registry_path = data_root / _DATASET_REGISTRY_RELATIVE_PATH
    if not registry_path.is_file():
        raise ReleaseValidationError([f"Dataset registry not found: {registry_path}"])
    registry = load_dataset_registry(registry_path)

    dataset_archives: dict[str, list[ArchiveInfo]] = {}
    noise_models_archive: ArchiveInfo | None = None

    zip_paths = sorted(staging_dir.glob("*.zip"), key=lambda path: path.name.casefold())
    if not zip_paths:
        raise ReleaseValidationError([f"No ZIP archives found in staging directory: {staging_dir}"])

    for zip_path in zip_paths:
        if zip_path.name.casefold() == "application.zip":
            continue
        if zip_path.name.lower().endswith(".lisai.zip"):
            continue
        try:
            archive = _inspect_zip(zip_path)
        except ReleaseValidationError as exc:
            errors.extend(exc.errors)
            continue

        if archive.top_level == _NOISE_MODELS_TOP_LEVEL:
            if noise_models_archive is not None:
                errors.append(
                    "Multiple noise-model archives found: "
                    f"{noise_models_archive.path.name!r} and {archive.path.name!r}."
                )
            else:
                noise_models_archive = archive
            continue
        if archive.top_level in _IGNORED_ZIP_TOP_LEVELS:
            continue
        if archive.top_level not in registry:
            errors.append(
                f"Archive {zip_path.name!r} has top-level directory {archive.top_level!r}, "
                "which is not a registered dataset and is not a recognized standalone "
                "release archive."
            )
            continue
        dataset_archives.setdefault(archive.top_level, []).append(archive)

    if not dataset_archives:
        errors.append("No dataset archives matching the dataset registry were found in staging.")

    dependencies: dict[str, DatasetDependencies] = {}
    sorted_dataset_archives = sorted(
        dataset_archives.items(),
        key=lambda item: item[0].casefold(),
    )
    for dataset_name, archives in sorted_dataset_archives:
        info = registry[dataset_name]
        usage = str(info.get("usage") or "training").strip().lower()
        if usage not in {"training", "evaluation"}:
            errors.append(
                f"Dataset {dataset_name!r} has unsupported usage {usage!r}; "
                "expected 'training' or 'evaluation'."
            )
            continue
        errors.extend(
            _validate_dataset_archives(
                dataset_name=dataset_name,
                archives=archives,
            )
        )
        try:
            dependencies[dataset_name] = _discover_dependencies(archives)
        except ReleaseValidationError as exc:
            errors.extend(exc.errors)

    staged_dataset_names = set(dataset_archives)
    required_noise_models: set[str] = set()
    for dataset_name, deps in dependencies.items():
        required_noise_models.update(deps.noise_models)
        for evaluation_name in deps.evaluation_datasets:
            registry_info = registry.get(evaluation_name)
            if registry_info is None:
                errors.append(
                    f"Dataset {dataset_name!r} has a saved evaluation referencing "
                    f"unregistered dataset {evaluation_name!r}."
                )
                continue
            eval_usage = str(registry_info.get("usage") or "training").strip().lower()
            if eval_usage != "evaluation":
                errors.append(
                    f"Dataset {dataset_name!r} has a saved evaluation referencing "
                    f"{evaluation_name!r}, but its registry usage is {eval_usage!r}, "
                    "not 'evaluation'."
                )
                continue
            if evaluation_name not in staged_dataset_names:
                errors.append(
                    f"Dataset {dataset_name!r} has a saved evaluation referencing "
                    f"{evaluation_name!r}, but that evaluation dataset has no staged archive."
                )

    errors.extend(
        _validate_noise_models(
            archive=noise_models_archive,
            required=required_noise_models,
        )
    )
    if noise_models_archive is not None and not required_noise_models:
        warnings.append(
            f"{noise_models_archive.path.name!r} is staged, but no staged dataset references "
            "a noise model."
        )

    if errors:
        raise ReleaseValidationError(errors)

    if progress:
        print("Release validation passed. Computing archive checksums...", file=stream)

    datasets: dict[str, Any] = {}
    for dataset_name in sorted(dataset_archives, key=str.casefold):
        info = registry[dataset_name]
        usage = str(info.get("usage") or "training").strip().lower()
        deps = dependencies[dataset_name]
        archives = sorted(
            dataset_archives[dataset_name],
            key=lambda archive: archive.path.name.casefold(),
        )
        datasets[dataset_name] = {
            "usage": usage,
            "install_path": f"datasets/{usage}",
            "archives": [
                _archive_record(archive.path, progress=progress, stream=stream)
                for archive in archives
            ],
            "registry_entry": info,
            "dependencies": {
                "evaluation_datasets": list(deps.evaluation_datasets),
                "noise_models": list(deps.noise_models),
            },
        }

    shared: dict[str, Any] = {}
    if noise_models_archive is not None:
        shared["noise_models"] = {
            "archive": _archive_record(
                noise_models_archive.path,
                progress=progress,
                stream=stream,
            ),
            "available": sorted(
                _noise_model_dirs_in_archive(noise_models_archive),
                key=str.casefold,
            ),
        }

    catalog: dict[str, Any] = {
        "schema_version": DATASET_CATALOG_SCHEMA_VERSION,
        "datasets": datasets,
    }
    if shared:
        catalog["shared"] = shared

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)

    if progress:
        print("", file=stream)
        print(f"Catalog written: {output_path}", file=stream)
        print(f"Datasets: {len(datasets)}", file=stream)
        total_size = sum(
            archive["size_bytes"]
            for dataset in datasets.values()
            for archive in dataset["archives"]
        )
        print(f"Dataset archives: {_format_bytes(total_size)}", file=stream)
        if required_noise_models:
            print(
                "Required noise models: "
                + ", ".join(sorted(required_noise_models, key=str.casefold)),
                file=stream,
            )
        for warning in warnings:
            print(f"WARNING: {warning}", file=stream)

    return catalog


__all__ = [
    "DATASET_CATALOG_SCHEMA_VERSION",
    "ReleaseValidationError",
    "build_dataset_catalog",
]
