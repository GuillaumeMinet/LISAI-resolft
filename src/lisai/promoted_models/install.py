from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from lisai.config import settings
from lisai.infra.paths import Paths

from .package import MODEL_MANIFEST_FILENAME, PromotedModel, load_promoted_model_from_dir, sha256_file
from .registry import load_promoted_model_registry, register_promoted_model
from .schema import PromotedModelManifest, PromotedModelRegistryEntry


@dataclass(frozen=True)
class PromotedModelInstall:
    model: PromotedModel
    archive_path: Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalized_archive_member(name: str) -> PurePosixPath:
    normalized = str(name).replace("\\", "/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    try:
        archive = zipfile.ZipFile(archive_path, mode="r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid promoted-model ZIP archive: {archive_path}") from exc

    destination_root = destination.resolve()
    seen_members: set[str] = set()
    with archive:
        for info in archive.infolist():
            member = _normalized_archive_member(info.filename)
            member_key = member.as_posix().rstrip("/")
            if member_key in seen_members:
                raise ValueError(f"Archive contains duplicate member path: {info.filename!r}")
            seen_members.add(member_key)
            if _is_zip_symlink(info):
                raise ValueError(f"Archive contains unsupported symbolic link: {info.filename!r}")
            target = destination.joinpath(*member.parts)
            target_resolved = target.resolve()
            if not target_resolved.is_relative_to(destination_root):
                raise ValueError(f"Unsafe archive member path: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _declared_artifact_paths(manifest: PromotedModelManifest) -> set[str]:
    paths = {manifest.artifacts.config, manifest.artifacts.weights}
    if manifest.artifacts.loss is not None:
        paths.add(manifest.artifacts.loss)
    if manifest.artifacts.loss_plot is not None:
        paths.add(manifest.artifacts.loss_plot)
    if manifest.artifacts.noise_model is not None:
        paths.add(manifest.artifacts.noise_model.model)
        paths.add(manifest.artifacts.noise_model.norm_prm)
    return paths


def validate_installable_model_dir(model_dir: str | Path) -> PromotedModel:
    """Validate an extracted promoted-model package without instantiating its runtime."""
    model_dir = Path(model_dir)
    promoted = load_promoted_model_from_dir(model_dir)
    manifest = promoted.manifest

    declared = _declared_artifact_paths(manifest)
    missing = sorted(path for path in declared if not (model_dir / path).is_file())
    if missing:
        raise FileNotFoundError(
            "Promoted-model archive is incomplete; missing declared artifact(s):\n"
            + "\n".join(missing)
        )

    undeclared_checksums = sorted(set(manifest.checksums) - declared)
    if undeclared_checksums:
        raise ValueError(
            "Promoted-model manifest contains checksums for undeclared artifact(s): "
            + ", ".join(undeclared_checksums)
        )

    missing_checksums = sorted(declared - set(manifest.checksums))
    if missing_checksums:
        raise ValueError(
            "Promoted-model manifest is missing SHA256 checksum(s) for: "
            + ", ".join(missing_checksums)
        )

    mismatches: list[str] = []
    for package_path in sorted(declared):
        expected = manifest.checksums[package_path]
        actual = sha256_file(model_dir / package_path)
        if actual != expected:
            mismatches.append(package_path)
    if mismatches:
        raise ValueError(
            "Promoted-model archive failed SHA256 verification for: " + ", ".join(mismatches)
        )
    return promoted


def install_model_archive(
    archive_path: str | Path,
    *,
    overwrite: bool = False,
    paths: Paths | None = None,
) -> PromotedModelInstall:
    """Install a local .lisai.zip archive into the canonical promoted-model library."""
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Promoted-model archive not found: {archive_path}")
    if not archive_path.name.endswith(".lisai.zip"):
        raise ValueError("Promoted-model archives must use the '.lisai.zip' suffix.")

    resolved_paths = paths or Paths(settings)
    promoted_root = resolved_paths.promoted_models_root()
    promoted_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lisai-install-", dir=promoted_root) as tmp:
        staging_dir = Path(tmp)
        _safe_extract_archive(archive_path, staging_dir)
        manifest_path = staging_dir / MODEL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Promoted-model archive must contain {MODEL_MANIFEST_FILENAME!r} at its root."
            )
        promoted = validate_installable_model_dir(staging_dir)
        name = promoted.manifest.name
        model_dir = resolved_paths.promoted_model_dir(model_name=name)
        exports_dir = resolved_paths.promoted_model_exports_dir()
        registry_path = resolved_paths.promoted_model_registry_path()
        if model_dir.resolve() in {exports_dir.resolve(), registry_path.resolve()}:
            raise ValueError(f"Promoted-model name {name!r} is reserved by LISAI.")

        registry = load_promoted_model_registry(paths=resolved_paths)
        existing_entry = registry.models.get(name)
        stale_registry_entry = False
        if existing_entry is not None:
            registered_dir = promoted_root / existing_entry.path
            stale_registry_entry = not registered_dir.is_dir()
            if not overwrite and not stale_registry_entry:
                raise FileExistsError(
                    f"Promoted model {name!r} is already registered. Use --overwrite to replace it."
                )
        if model_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Promoted-model directory already exists: {model_dir}. Use --overwrite to replace it."
            )

        # Validation is complete before any existing installation is touched.
        backup_dir: Path | None = None
        try:
            if model_dir.exists():
                backup_dir = model_dir.with_name(f".{model_dir.name}.install-backup")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                os.replace(model_dir, backup_dir)
            os.replace(staging_dir, model_dir)
            register_promoted_model(
                name=name,
                entry=PromotedModelRegistryEntry(
                    source_run_id=promoted.manifest.source.run_id,
                    path=model_dir.relative_to(promoted_root).as_posix(),
                    created_at=promoted.manifest.created_at,
                    origin="installed",
                    installed_at=_utc_now(),
                    task=promoted.manifest.model.task,
                ),
                overwrite=overwrite or stale_registry_entry,
                paths=resolved_paths,
            )
        except Exception:
            if model_dir.exists():
                shutil.rmtree(model_dir)
            if backup_dir is not None and backup_dir.exists():
                os.replace(backup_dir, model_dir)
            raise
        else:
            if backup_dir is not None and backup_dir.exists():
                shutil.rmtree(backup_dir)

    return PromotedModelInstall(
        model=load_promoted_model_from_dir(model_dir),
        archive_path=archive_path,
    )


__all__ = [
    "PromotedModelInstall",
    "install_model_archive",
    "validate_installable_model_dir",
]
