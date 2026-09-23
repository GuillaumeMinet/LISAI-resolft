from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from lisai.data.catalog import DatasetCatalog
from lisai.data.dataset_registry import load_dataset_registry, save_dataset_registry
from lisai.data.download import download_and_install_dataset_plan, download_summary
from lisai.data.install import (
    DatasetInstallConflictError,
    DatasetInstallSource,
    discover_install_source,
    install_dataset_plan,
    missing_evaluation_dependencies,
    resolve_dataset_plan,
    validate_install_source,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zip_dataset(path: Path, name: str, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative, data in files.items():
            archive.writestr(f"{name}/{relative}", data)


def _zip_noise_models(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("noise_models/noise_a/GMM.npz", b"gmm")
        archive.writestr("noise_models/noise_a/norm_prm.json", b"{}")


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def data_root(self) -> Path:
        return self.root

    def datasets_root(self) -> Path:
        return self.root / "datasets"

    def dataset_registry_path(self) -> Path:
        return self.root / "datasets" / "dataset_registry.yml"

    def dataset_dir(self, *, dataset_name: str, usage: str = "training", data_subfolder: str = "") -> Path:
        return self.root / "datasets" / usage / dataset_name / data_subfolder

    def noise_model_dir(self, *, noiseModel_name: str) -> Path:
        return self.root / "noise_models" / noiseModel_name


def _catalog(train_zip: Path, eval_zip: Path, noise_zip: Path) -> DatasetCatalog:
    return DatasetCatalog.model_validate(
        {
            "schema_version": 1,
            "datasets": {
                "train_a": {
                    "usage": "training",
                    "install_path": "datasets/training",
                    "archives": [
                        {
                            "filename": train_zip.name,
                            "size_bytes": train_zip.stat().st_size,
                            "sha256": _sha(train_zip),
                        }
                    ],
                    "registry_entry": {
                        "usage": "training",
                        "data_format": "single",
                    },
                    "dependencies": {
                        "evaluation_datasets": ["eval_a"],
                        "noise_models": ["noise_a"],
                    },
                },
                "eval_a": {
                    "usage": "evaluation",
                    "install_path": "datasets/evaluation",
                    "archives": [
                        {
                            "filename": eval_zip.name,
                            "size_bytes": eval_zip.stat().st_size,
                            "sha256": _sha(eval_zip),
                        }
                    ],
                    "registry_entry": {
                        "usage": "evaluation",
                        "data_format": "single",
                    },
                    "dependencies": {
                        "evaluation_datasets": [],
                        "noise_models": [],
                    },
                },
            },
            "shared": {
                "noise_models": {
                    "archive": {
                        "filename": noise_zip.name,
                        "size_bytes": noise_zip.stat().st_size,
                        "sha256": _sha(noise_zip),
                    },
                    "available": ["noise_a"],
                }
            },
        }
    )


def _release_files(tmp_path: Path):
    release = tmp_path / "release"
    release.mkdir()
    train_zip = release / "train_a.zip"
    eval_zip = release / "eval_a.zip"
    noise_zip = release / "noise_models.zip"
    _zip_dataset(train_zip, "train_a", {"raw/a.tif": b"train"})
    _zip_dataset(eval_zip, "eval_a", {"raw/b.tif": b"eval"})
    _zip_noise_models(noise_zip)
    return release, train_zip, eval_zip, noise_zip


def test_shared_plan_reports_optional_eval_and_required_noise_models(tmp_path: Path):
    _release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")

    assert missing_evaluation_dependencies(catalog, ["train_a"], paths=paths) == ["eval_a"]

    plan = resolve_dataset_plan(catalog, ["train_a"], paths=paths)

    assert [entry.name for entry in plan.entries] == ["train_a"]
    assert plan.required_noise_models == ("noise_a",)
    assert plan.dataset_install_bytes == train_zip.stat().st_size
    assert download_summary(catalog, plan).total_bytes == (
        train_zip.stat().st_size + noise_zip.stat().st_size
    )


def test_local_install_extracts_dataset_noise_model_and_merges_registry(tmp_path: Path):
    _release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")
    plan = resolve_dataset_plan(catalog, ["train_a", "eval_a"], paths=paths)
    source = DatasetInstallSource(
        archives={train_zip.name: train_zip, eval_zip.name: eval_zip},
        noise_models_archive=noise_zip,
    )

    result = install_dataset_plan(
        catalog,
        plan,
        source=source,
        paths=paths,
        stream=io.StringIO(),
    )

    assert result.installed == ("eval_a", "train_a")
    assert (paths.dataset_dir(dataset_name="train_a") / "raw" / "a.tif").read_bytes() == b"train"
    assert (
        paths.dataset_dir(dataset_name="eval_a", usage="evaluation") / "raw" / "b.tif"
    ).read_bytes() == b"eval"
    assert (paths.noise_model_dir(noiseModel_name="noise_a") / "GMM.npz").is_file()
    registry = load_dataset_registry(paths.dataset_registry_path())
    assert set(registry) == {"train_a", "eval_a"}


def test_download_fetches_then_uses_common_installer(monkeypatch, tmp_path: Path):
    _release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")
    plan = resolve_dataset_plan(catalog, ["train_a", "eval_a"], paths=paths)

    archive_lookup = {
        train_zip.name: train_zip,
        eval_zip.name: eval_zip,
        noise_zip.name: noise_zip,
    }

    def fake_download(archive, *, url, downloads_dir, timeout, retries, stream):
        return archive_lookup[archive.filename]

    captured = {}

    def fake_install(catalog_arg, plan_arg, *, source, paths, stream):
        captured["source"] = source
        from lisai.data.install import DatasetInstallResult

        return DatasetInstallResult(
            installed=("eval_a", "train_a"),
            already_installed=(),
            noise_models_installed=("noise_a",),
        )

    monkeypatch.setattr("lisai.data.download._download_archive", fake_download)
    monkeypatch.setattr("lisai.data.download.install_dataset_plan", fake_install)

    result = download_and_install_dataset_plan(
        catalog,
        plan,
        record_id="123",
        paths=paths,
        stream=io.StringIO(),
        url_resolver=lambda filename: f"https://example.org/{filename}",
    )

    assert result.installed == ("eval_a", "train_a")
    assert set(captured["source"].archives) == {"train_a.zip", "eval_a.zip"}
    assert captured["source"].noise_models_archive == noise_zip


def test_existing_dataset_still_repairs_missing_noise_model(tmp_path: Path):
    _release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")
    dataset_dir = paths.dataset_dir(dataset_name="train_a")
    dataset_dir.mkdir(parents=True)
    save_dataset_registry(
        {"train_a": {"usage": "training", "data_format": "single"}},
        paths.dataset_registry_path(),
    )

    plan = resolve_dataset_plan(catalog, ["train_a"], paths=paths)

    assert plan.entries[0].status == "installed"
    assert plan.required_noise_models == ("noise_a",)
    assert plan.dataset_install_bytes == 0


def test_installed_noise_model_does_not_require_noise_archive(tmp_path: Path):
    release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")
    noise_dir = paths.noise_model_dir(noiseModel_name="noise_a")
    noise_dir.mkdir(parents=True)
    (noise_dir / "GMM.npz").write_bytes(b"existing")
    (noise_dir / "norm_prm.json").write_text("{}")

    plan = resolve_dataset_plan(catalog, ["train_a"], paths=paths)
    selected, source = discover_install_source(catalog, train_zip)

    assert selected == ["train_a"]
    assert plan.required_noise_models == ()
    source_without_noise = DatasetInstallSource(archives=source.archives)
    validate_install_source(catalog, plan, source_without_noise)


def test_installed_evaluation_dataset_is_not_reported_as_missing(tmp_path: Path):
    _release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")
    eval_dir = paths.dataset_dir(dataset_name="eval_a", usage="evaluation")
    eval_dir.mkdir(parents=True)
    save_dataset_registry(
        {"eval_a": {"usage": "evaluation", "data_format": "single"}},
        paths.dataset_registry_path(),
    )

    assert missing_evaluation_dependencies(catalog, ["train_a"], paths=paths) == []


def test_archive_path_selects_only_that_dataset_even_with_sibling_archives(tmp_path: Path):
    release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)

    selected, source = discover_install_source(catalog, train_zip)

    assert selected == ["train_a"]
    assert set(source.archives) == {"train_a.zip"}
    assert source.noise_models_archive == noise_zip


def test_directory_selects_all_complete_catalog_datasets(tmp_path: Path):
    release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    (release / "application.zip").write_bytes(b"ignored")
    (release / "model.lisai.zip").write_bytes(b"ignored")

    selected, source = discover_install_source(catalog, release)

    assert selected == ["eval_a", "train_a"]
    assert set(source.archives) == {"train_a.zip", "eval_a.zip"}
    assert source.noise_models_archive == noise_zip


def test_unregistered_existing_destination_is_a_conflict(tmp_path: Path):
    _release, train_zip, eval_zip, noise_zip = _release_files(tmp_path)
    catalog = _catalog(train_zip, eval_zip, noise_zip)
    paths = FakePaths(tmp_path / "data")
    paths.dataset_dir(dataset_name="train_a").mkdir(parents=True)

    with pytest.raises(DatasetInstallConflictError, match="not registered"):
        resolve_dataset_plan(catalog, ["train_a"], paths=paths)
