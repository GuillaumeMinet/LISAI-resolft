from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal, Mapping, TextIO

from lisai.config import settings
from lisai.infra.paths import Paths

from .catalog import DatasetArchive, DatasetCatalog, DownloadableDataset
from .dataset_registry import load_dataset_registry, save_dataset_registry


_IO_CHUNK_SIZE = 8 * 1024 * 1024
_REQUIRED_NOISE_MODEL_FILES = ("GMM.npz", "norm_prm.json")


class DatasetInstallError(RuntimeError):
    """Raised when a local dataset release cannot be installed safely."""


class DatasetInstallIntegrityError(DatasetInstallError):
    """Raised when a local release archive does not match the catalog."""


class DatasetInstallConflictError(DatasetInstallError):
    """Raised when installation would overwrite incompatible local state."""


@dataclass(frozen=True)
class DatasetPlanEntry:
    name: str
    dataset: DownloadableDataset
    status: Literal["install", "installed"]

    @property
    def size_bytes(self) -> int:
        return sum(archive.size_bytes for archive in self.dataset.archives)


@dataclass(frozen=True)
class DatasetInstallPlan:
    entries: tuple[DatasetPlanEntry, ...]
    required_noise_models: tuple[str, ...]

    @property
    def dataset_install_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries if entry.status == "install")

    @property
    def pending_dataset_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries if entry.status == "install")


@dataclass(frozen=True)
class DatasetInstallSource:
    """Local release files used by the installer.

    ``archives`` is keyed by the catalog filename, not by dataset name. This lets
    one logical dataset use multiple physical ZIP files without changing the
    installation API.
    """

    archives: Mapping[str, Path]
    noise_models_archive: Path | None = None
    verified_archives: frozenset[str] = frozenset()
    noise_models_verified: bool = False


@dataclass(frozen=True)
class DatasetInstallResult:
    installed: tuple[str, ...]
    already_installed: tuple[str, ...]
    noise_models_installed: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_IO_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(path: Path, archive: DatasetArchive) -> None:
    if not path.is_file():
        raise DatasetInstallError(f"Required archive is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != archive.size_bytes:
        raise DatasetInstallIntegrityError(
            f"Archive {path.name!r} has the wrong size: expected {archive.size_bytes} bytes, "
            f"got {actual_size}."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != archive.sha256:
        raise DatasetInstallIntegrityError(
            f"Archive {path.name!r} failed SHA256 verification: expected {archive.sha256}, "
            f"got {actual_sha256}."
        )


def _safe_members(archive: zipfile.ZipFile, *, expected_top_level: str) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/").lstrip("/")
        if not normalized:
            continue
        parts = PurePosixPath(normalized).parts
        if not parts or parts[0] != expected_top_level:
            raise DatasetInstallError(
                f"Archive member {info.filename!r} is outside expected top-level directory "
                f"{expected_top_level!r}."
            )
        if any(part in {"", ".", ".."} for part in parts):
            raise DatasetInstallError(f"Unsafe archive member path: {info.filename!r}")
        members.append(info)
    return members


def _extract_archive(
    archive_path: Path,
    *,
    destination: Path,
    expected_top_level: str,
    include_top_level_children: set[str] | None = None,
) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _safe_members(archive, expected_top_level=expected_top_level)
            destination.mkdir(parents=True, exist_ok=True)
            root = destination.resolve()
            for info in members:
                normalized = info.filename.replace("\\", "/").lstrip("/")
                parts = PurePosixPath(normalized).parts
                if include_top_level_children is not None:
                    if len(parts) < 2 or parts[1] not in include_top_level_children:
                        continue
                target = (destination / Path(*parts)).resolve()
                if target != root and root not in target.parents:
                    raise DatasetInstallError(f"Unsafe archive member path: {info.filename!r}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=_IO_CHUNK_SIZE)
    except zipfile.BadZipFile as exc:
        raise DatasetInstallError(f"Invalid ZIP archive {archive_path}: {exc}") from exc


def dataset_is_installed(name: str, dataset: DownloadableDataset, *, paths: Paths) -> bool:
    dataset_dir = paths.dataset_dir(dataset_name=name, usage=dataset.usage)
    registry = load_dataset_registry(paths.dataset_registry_path())
    return dataset_dir.is_dir() and name in registry and registry[name] == dataset.registry_entry


def noise_model_is_installed(name: str, *, paths: Paths) -> bool:
    noise_dir = paths.noise_model_dir(noiseModel_name=name)
    return all((noise_dir / filename).is_file() for filename in _REQUIRED_NOISE_MODEL_FILES)


def missing_evaluation_dependencies(
    catalog: DatasetCatalog,
    selected_names: Iterable[str],
    *,
    paths: Paths | None = None,
) -> list[str]:
    """Return optional evaluation dependencies not selected and not already installed."""
    resolved_paths = paths or Paths(settings)
    selected = set(selected_names)
    missing: set[str] = set()
    for name in selected:
        dataset = catalog.datasets[name]
        for dependency in dataset.dependencies.evaluation_datasets:
            if dependency in selected:
                continue
            dependency_entry = catalog.datasets.get(dependency)
            if dependency_entry is None:
                raise DatasetInstallError(
                    f"Dataset {name!r} references evaluation dataset {dependency!r}, "
                    "but it is absent from the dataset catalog."
                )
            if not dataset_is_installed(dependency, dependency_entry, paths=resolved_paths):
                missing.add(dependency)
    return sorted(missing, key=str.casefold)


def resolve_dataset_plan(
    catalog: DatasetCatalog,
    selected_names: Iterable[str],
    *,
    paths: Paths | None = None,
) -> DatasetInstallPlan:
    """Resolve installed state and hard dependencies for either download or local install."""
    resolved_paths = paths or Paths(settings)
    names = sorted(set(selected_names), key=str.casefold)
    entries: list[DatasetPlanEntry] = []
    required_noise_models: set[str] = set()

    local_registry = load_dataset_registry(resolved_paths.dataset_registry_path())
    for name in names:
        try:
            dataset = catalog.datasets[name]
        except KeyError as exc:
            raise DatasetInstallError(f"Unknown downloadable dataset {name!r}.") from exc

        dataset_dir = resolved_paths.dataset_dir(dataset_name=name, usage=dataset.usage)
        registered = name in local_registry
        if registered and local_registry[name] != dataset.registry_entry:
            raise DatasetInstallConflictError(
                f"Dataset {name!r} is already registered with metadata that differs from "
                "the dataset catalog."
            )
        if dataset_dir.exists() and not registered:
            raise DatasetInstallConflictError(
                f"Dataset destination already exists but is not registered: {dataset_dir}"
            )

        status: Literal["install", "installed"] = (
            "installed" if dataset_dir.is_dir() and registered else "install"
        )
        entries.append(DatasetPlanEntry(name=name, dataset=dataset, status=status))
        required_noise_models.update(dataset.dependencies.noise_models)

    missing_noise = tuple(
        sorted(
            (
                name
                for name in required_noise_models
                if not noise_model_is_installed(name, paths=resolved_paths)
            ),
            key=str.casefold,
        )
    )
    if missing_noise:
        bundle = catalog.shared.noise_models if catalog.shared else None
        if bundle is None:
            raise DatasetInstallError(
                "Selected datasets require noise models, but the catalog has no noise_models archive."
            )
        unavailable = [name for name in missing_noise if name not in bundle.available]
        if unavailable:
            raise DatasetInstallError(
                "Required noise model(s) are absent from the shared archive: "
                + ", ".join(unavailable)
            )

    return DatasetInstallPlan(
        entries=tuple(entries),
        required_noise_models=missing_noise,
    )


def _merge_registry_entry(
    name: str,
    dataset: DownloadableDataset,
    *,
    paths: Paths,
) -> None:
    registry_path = paths.dataset_registry_path()
    registry = load_dataset_registry(registry_path)
    existing = registry.get(name)
    if existing is not None and existing != dataset.registry_entry:
        raise DatasetInstallConflictError(
            f"Dataset {name!r} is already registered with different metadata in {registry_path}."
        )
    registry[name] = dict(dataset.registry_entry)
    save_dataset_registry(registry, registry_path)


def _required_archive_paths(
    plan: DatasetInstallPlan,
    source: DatasetInstallSource,
) -> dict[str, Path]:
    required: dict[str, Path] = {}
    for entry in plan.entries:
        if entry.status == "installed":
            continue
        for archive in entry.dataset.archives:
            path = source.archives.get(archive.filename)
            if path is None:
                raise DatasetInstallError(
                    f"Required archive {archive.filename!r} for dataset {entry.name!r} "
                    "is not available in the installation source."
                )
            required[archive.filename] = Path(path)
    return required


def validate_install_source(
    catalog: DatasetCatalog,
    plan: DatasetInstallPlan,
    source: DatasetInstallSource,
) -> None:
    """Cheap preflight: verify required local files exist before confirmation/hashing."""
    required = _required_archive_paths(plan, source)
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise DatasetInstallError("Required dataset archive(s) are missing: " + ", ".join(missing))

    if plan.required_noise_models:
        if catalog.shared is None or catalog.shared.noise_models is None:
            raise DatasetInstallError(
                "Selected datasets require noise models, but the catalog has no noise_models archive."
            )
        if source.noise_models_archive is None or not source.noise_models_archive.is_file():
            required_names = ", ".join(plan.required_noise_models)
            raise DatasetInstallError(
                f"Required noise model(s) are not installed ({required_names}) and "
                "noise_models.zip is not available in the installation source."
            )


def install_dataset_plan(
    catalog: DatasetCatalog,
    plan: DatasetInstallPlan,
    *,
    source: DatasetInstallSource,
    paths: Paths | None = None,
    stream: TextIO | None = None,
) -> DatasetInstallResult:
    """Install a resolved plan exclusively from local, catalogued archives."""
    resolved_paths = paths or Paths(settings)
    output = sys.stdout if stream is None else stream
    validate_install_source(catalog, plan, source)

    installed: list[str] = []
    already_installed = [entry.name for entry in plan.entries if entry.status == "installed"]
    noise_models_installed: list[str] = []

    if plan.required_noise_models:
        assert catalog.shared is not None and catalog.shared.noise_models is not None
        assert source.noise_models_archive is not None
        bundle = catalog.shared.noise_models
        archive_path = source.noise_models_archive
        if not source.noise_models_verified:
            print(f"\nVerifying {archive_path.name}...", file=output)
            _verify_archive(archive_path, bundle.archive)
        print("Extracting required noise models...", file=output)
        _extract_archive(
            archive_path,
            destination=resolved_paths.data_root(),
            expected_top_level="noise_models",
            include_top_level_children=set(plan.required_noise_models),
        )
        for name in plan.required_noise_models:
            if not noise_model_is_installed(name, paths=resolved_paths):
                raise DatasetInstallError(
                    f"Noise model {name!r} is still incomplete after extracting "
                    f"{archive_path.name!r}."
                )
            noise_models_installed.append(name)

    for entry in plan.entries:
        if entry.status == "installed":
            continue

        print(f"\nDataset: {entry.name}", file=output)
        destination = resolved_paths.dataset_dir(
            dataset_name=entry.name,
            usage=entry.dataset.usage,
        )
        if destination.exists():
            raise DatasetInstallConflictError(
                f"Dataset destination appeared during installation and will not be overwritten: "
                f"{destination}"
            )

        try:
            for archive in entry.dataset.archives:
                archive_path = Path(source.archives[archive.filename])
                if archive.filename not in source.verified_archives:
                    print(f"Verifying {archive.filename}...", file=output)
                    _verify_archive(archive_path, archive)
                print(f"Extracting {archive.filename}...", file=output)
                _extract_archive(
                    archive_path,
                    destination=destination.parent,
                    expected_top_level=entry.name,
                )

            if not destination.is_dir():
                raise DatasetInstallError(
                    f"Dataset {entry.name!r} was not created at expected path {destination}."
                )
            _merge_registry_entry(entry.name, entry.dataset, paths=resolved_paths)
            installed.append(entry.name)
        except Exception:
            # The destination did not exist when this installation started, so any
            # directory now present is an incomplete extraction owned by this attempt.
            shutil.rmtree(destination, ignore_errors=True)
            raise

    return DatasetInstallResult(
        installed=tuple(installed),
        already_installed=tuple(already_installed),
        noise_models_installed=tuple(noise_models_installed),
    )


def _catalog_archive_owners(catalog: DatasetCatalog) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for name, dataset in catalog.datasets.items():
        for archive in dataset.archives:
            owners.setdefault(archive.filename, []).append(name)
    return owners


def _zip_top_levels(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            roots: set[str] = set()
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/").lstrip("/")
                if normalized:
                    roots.add(PurePosixPath(normalized).parts[0])
            return roots
    except zipfile.BadZipFile as exc:
        raise DatasetInstallError(f"Invalid ZIP archive {path}: {exc}") from exc


def _dataset_for_archive_path(catalog: DatasetCatalog, archive_path: Path) -> str:
    owners = _catalog_archive_owners(catalog).get(archive_path.name, [])
    if len(owners) == 1:
        return owners[0]
    if len(owners) > 1:
        raise DatasetInstallError(
            f"Archive filename {archive_path.name!r} belongs to multiple catalog datasets: "
            + ", ".join(sorted(owners, key=str.casefold))
        )

    # Friendly fallback for a manually renamed single-archive ZIP: identify it by
    # its required dataset top-level directory. The checksum still validates the
    # actual bytes during installation.
    top_levels = _zip_top_levels(archive_path)
    matches = [
        name
        for name, dataset in catalog.datasets.items()
        if len(dataset.archives) == 1 and top_levels == {name}
    ]
    if len(matches) == 1:
        return matches[0]

    raise DatasetInstallError(
        f"Could not match archive {archive_path.name!r} to a dataset in the catalog."
    )


def discover_install_source(
    catalog: DatasetCatalog,
    source_path: str | Path,
) -> tuple[list[str], DatasetInstallSource]:
    """Discover dataset selection and local archive paths from a ZIP or directory.

    A ZIP path selects exactly that logical dataset. A directory selects every
    complete catalog dataset whose archive(s) are present in the directory.
    Unrelated files (application.zip, promoted models, etc.) are ignored.
    """
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise DatasetInstallError(f"Installation source does not exist: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() != ".zip":
            raise DatasetInstallError(
                f"Dataset installation source must be a ZIP archive or directory: {source_path}"
            )
        dataset_name = _dataset_for_archive_path(catalog, source_path)
        dataset = catalog.datasets[dataset_name]
        archives: dict[str, Path] = {}
        for archive in dataset.archives:
            candidate = source_path if archive.filename == source_path.name else source_path.parent / archive.filename
            if not candidate.is_file() and len(dataset.archives) == 1:
                # Allow a renamed single-archive dataset identified by top-level folder.
                candidate = source_path
            archives[archive.filename] = candidate
        noise_archive = None
        if catalog.shared and catalog.shared.noise_models:
            candidate = source_path.parent / catalog.shared.noise_models.archive.filename
            if candidate.is_file():
                noise_archive = candidate
        return [dataset_name], DatasetInstallSource(
            archives=archives,
            noise_models_archive=noise_archive,
        )

    if not source_path.is_dir():
        raise DatasetInstallError(f"Unsupported installation source: {source_path}")

    selected: list[str] = []
    archives: dict[str, Path] = {}
    partial: list[tuple[str, list[str]]] = []
    for name in sorted(catalog.datasets, key=str.casefold):
        dataset = catalog.datasets[name]
        present = [
            archive
            for archive in dataset.archives
            if (source_path / archive.filename).is_file()
        ]
        if not present:
            continue
        missing = [
            archive.filename
            for archive in dataset.archives
            if not (source_path / archive.filename).is_file()
        ]
        if missing:
            partial.append((name, missing))
            continue
        selected.append(name)
        for archive in dataset.archives:
            archives[archive.filename] = source_path / archive.filename

    if partial:
        details = "; ".join(
            f"{name}: missing {', '.join(missing)}" for name, missing in partial
        )
        raise DatasetInstallError(
            "Incomplete multi-archive dataset(s) in installation directory: " + details
        )
    if not selected:
        raise DatasetInstallError(
            f"No dataset archives from the catalog were found in {source_path}."
        )

    noise_archive = None
    if catalog.shared and catalog.shared.noise_models:
        candidate = source_path / catalog.shared.noise_models.archive.filename
        if candidate.is_file():
            noise_archive = candidate

    return selected, DatasetInstallSource(
        archives=archives,
        noise_models_archive=noise_archive,
    )


__all__ = [
    "DatasetInstallConflictError",
    "DatasetInstallError",
    "DatasetInstallIntegrityError",
    "DatasetInstallPlan",
    "DatasetInstallResult",
    "DatasetInstallSource",
    "DatasetPlanEntry",
    "dataset_is_installed",
    "discover_install_source",
    "install_dataset_plan",
    "missing_evaluation_dependencies",
    "noise_model_is_installed",
    "resolve_dataset_plan",
    "validate_install_source",
]
