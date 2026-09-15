from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from lisai.promoted_models.install import install_model_archive
from lisai.promoted_models.schema import (
    PromotedModelArtifacts,
    PromotedModelDefinition,
    PromotedModelManifest,
    PromotedModelSource,
    PromotedTrainingData,
)


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def promoted_models_root(self):
        return self.root / "models"

    def promoted_model_registry_path(self):
        return self.promoted_models_root() / "model_registry.yml"

    def promoted_model_dir(self, *, model_name: str):
        return self.promoted_models_root() / model_name

    def promoted_model_exports_dir(self):
        return self.promoted_models_root() / "_exports"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_archive(tmp_path: Path, *, name: str = "demo-model") -> Path:
    source = tmp_path / "archive-source"
    source.mkdir()
    (source / "weights.pt").write_bytes(b"weights")
    (source / "config_train.yaml").write_text("not: necessarily-current-schema\n", encoding="utf-8")
    training = source / "training"
    training.mkdir()
    (training / "loss.txt").write_text("loss history\n", encoding="utf-8")

    manifest = PromotedModelManifest(
        name=name,
        created_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
        model=PromotedModelDefinition(task="denoising", architecture="unet"),
        training_data=PromotedTrainingData(dataset="DemoData"),
        source=PromotedModelSource(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            run_name="run_a",
            run_status="completed",
            checkpoint_selector="best",
            checkpoint_filename="model_best_state_dict.pt",
        ),
        artifacts=PromotedModelArtifacts(loss="training/loss.txt"),
        checksums={
            "weights.pt": _sha(source / "weights.pt"),
            "config_train.yaml": _sha(source / "config_train.yaml"),
            "training/loss.txt": _sha(training / "loss.txt"),
        },
    )
    (source / "lisai_model.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json", exclude_none=True), sort_keys=False),
        encoding="utf-8",
    )
    (source / "README.md").write_text("# Demo\n", encoding="utf-8")

    archive_path = tmp_path / f"{name}.lisai.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(source).as_posix())
    return archive_path


def test_install_archive_registers_same_canonical_model_representation(tmp_path: Path):
    archive_path = _make_archive(tmp_path)
    paths = FakePaths(tmp_path / "destination")

    installed = install_model_archive(archive_path, paths=paths)

    assert installed.model.manifest.name == "demo-model"
    assert installed.model.model_dir == paths.promoted_model_dir(model_name="demo-model")
    assert (installed.model.model_dir / "weights.pt").read_bytes() == b"weights"
    registry = yaml.safe_load(paths.promoted_model_registry_path().read_text())
    entry = registry["models"]["demo-model"]
    assert entry["origin"] == "installed"
    assert entry["source_run_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert entry["installed_at"] is not None


def test_install_does_not_require_runtime_config_to_validate(tmp_path: Path):
    archive_path = _make_archive(tmp_path)
    paths = FakePaths(tmp_path / "destination")

    installed = install_model_archive(archive_path, paths=paths)

    # The fixture's config is intentionally not a ResolvedExperiment. Installation
    # validates portability/integrity; runtime compatibility remains an apply-time concern.
    assert installed.model.model_dir.is_dir()


def test_install_rejects_checksum_mismatch_without_registering(tmp_path: Path):
    archive_path = _make_archive(tmp_path)
    tampered = tmp_path / "tampered.lisai.zip"
    with zipfile.ZipFile(archive_path, "r") as src, zipfile.ZipFile(tampered, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "weights.pt":
                data = b"tampered"
            dst.writestr(info, data)
    paths = FakePaths(tmp_path / "destination")

    with pytest.raises(ValueError, match="SHA256"):
        install_model_archive(tampered, paths=paths)

    assert not paths.promoted_model_registry_path().exists()
    assert not paths.promoted_model_dir(model_name="demo-model").exists()


def test_install_rejects_unsafe_zip_member(tmp_path: Path):
    archive_path = tmp_path / "unsafe.lisai.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", b"bad")
    paths = FakePaths(tmp_path / "destination")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        install_model_archive(archive_path, paths=paths)

    assert not (tmp_path / "outside.txt").exists()


def test_install_refuses_duplicate_name_unless_overwrite(tmp_path: Path):
    archive_path = _make_archive(tmp_path)
    paths = FakePaths(tmp_path / "destination")
    install_model_archive(archive_path, paths=paths)

    with pytest.raises(FileExistsError, match="--overwrite"):
        install_model_archive(archive_path, paths=paths)

    # An explicitly requested overwrite replaces the local snapshot and remains installed.
    installed = install_model_archive(archive_path, paths=paths, overwrite=True)
    assert installed.model.manifest.name == "demo-model"
    registry = yaml.safe_load(paths.promoted_model_registry_path().read_text())
    assert registry["models"]["demo-model"]["origin"] == "installed"


def test_install_requires_lisai_zip_suffix(tmp_path: Path):
    archive_path = tmp_path / "model.zip"
    archive_path.write_bytes(b"not-used")
    paths = FakePaths(tmp_path / "destination")

    with pytest.raises(ValueError, match=r"\.lisai\.zip"):
        install_model_archive(archive_path, paths=paths)


def test_install_accepts_registry_with_empty_models_value(tmp_path: Path):
    archive_path = _make_archive(tmp_path)
    paths = FakePaths(tmp_path / "destination")
    paths.promoted_model_registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.promoted_model_registry_path().write_text(
        "schema_version: 1\nmodels:\n",
        encoding="utf-8",
    )

    installed = install_model_archive(archive_path, paths=paths)

    assert installed.model.manifest.name == "demo-model"
    registry = yaml.safe_load(paths.promoted_model_registry_path().read_text())
    assert registry["models"]["demo-model"]["origin"] == "installed"


def test_install_repairs_stale_registry_entry_when_model_directory_is_missing(tmp_path: Path):
    archive_path = _make_archive(tmp_path)
    paths = FakePaths(tmp_path / "destination")
    installed = install_model_archive(archive_path, paths=paths)
    model_dir = installed.model.model_dir
    import shutil
    shutil.rmtree(model_dir)

    # The registry entry intentionally remains, mimicking manual removal of the
    # promoted-model directory before reinstalling a downloaded export.
    reinstalled = install_model_archive(archive_path, paths=paths)

    assert reinstalled.model.model_dir.is_dir()
    registry = yaml.safe_load(paths.promoted_model_registry_path().read_text())
    assert registry["models"]["demo-model"]["origin"] == "installed"
