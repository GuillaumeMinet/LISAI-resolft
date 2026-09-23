from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from lisai.data.release_builder import ReleaseValidationError, build_dataset_catalog


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _zip_folder(source: Path, destination: Path, *, root_name: str | None = None) -> None:
    root_name = root_name or source.name
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                archive.write(path, f"{root_name}/{relative}")


def _registry(data_root: Path, entries: dict) -> None:
    _write_yaml(data_root / "datasets" / "dataset_registry.yml", entries)


def test_build_dataset_catalog_discovers_only_staged_dependencies_and_allows_curated_content(
    tmp_path: Path,
):
    data_root = tmp_path / "lisai"
    staging = tmp_path / "staging"
    staging.mkdir()

    train_name = "vim_fixed_multi_snr"
    eval_name = "gag_live_upsamp_eval"
    _registry(
        data_root,
        {
            train_name: {
                "usage": "training",
                "data_format": "mltpl_snr",
                "description": "training data",
            },
            eval_name: {
                "usage": "evaluation",
                "data_format": "single",
                "description": "evaluation data",
            },
        },
    )

    # The local source is intentionally NOT identical to the staged release.
    # It contains a historical run that must not affect release dependencies.
    source_train_dir = data_root / "datasets" / "training" / train_name
    source_train_dir.mkdir(parents=True)
    (source_train_dir / "source_only.tif").write_bytes(b"not-released")
    _write_yaml(
        source_train_dir
        / "runs"
        / "old"
        / "evaluations"
        / "legacy"
        / "best"
        / "evaluation.yaml",
        {"dataset": {"name": "TestDataset_Gag_low_high", "usage": "evaluation"}},
    )

    # Build a curated release copy with selected data/runs plus a release-only file.
    release_train_dir = tmp_path / "release_train" / train_name
    (release_train_dir / "preprocess").mkdir(parents=True)
    (release_train_dir / "preprocess" / "sample.tif").write_bytes(b"training-data")
    (release_train_dir / "RELEASE_NOTES.txt").write_text("curated release", encoding="utf-8")
    _write_yaml(
        release_train_dir / "runs" / "hdn" / "run_01" / "config_train.yaml",
        {"noise_model": {"name": "monalisa_noise"}},
    )
    _write_yaml(
        release_train_dir
        / "runs"
        / "hdn"
        / "run_01"
        / "evaluations"
        / eval_name
        / "best"
        / "evaluation.yaml",
        {"dataset": {"name": eval_name, "usage": "evaluation"}},
    )
    # Training-split evaluations are not external dataset dependencies.
    _write_yaml(
        release_train_dir
        / "runs"
        / "hdn"
        / "run_01"
        / "evaluations"
        / "training_test"
        / "best"
        / "evaluation.yaml",
        {"dataset": {"name": train_name, "usage": "training", "split": "test"}},
    )

    release_eval_dir = tmp_path / "release_eval" / eval_name
    release_eval_dir.mkdir(parents=True)
    (release_eval_dir / "sample.tif").write_bytes(b"evaluation-data")

    release_noise_dir = tmp_path / "release_noise" / "noise_models" / "monalisa_noise"
    release_noise_dir.mkdir(parents=True)
    (release_noise_dir / "GMM.npz").write_bytes(b"gmm")
    (release_noise_dir / "norm_prm.json").write_text("{}", encoding="utf-8")

    train_zip = staging / "training_vimentin.zip"
    eval_zip = staging / "evaluation_gag.zip"
    noise_zip = staging / "noise_models.zip"
    _zip_folder(release_train_dir, train_zip, root_name=train_name)
    _zip_folder(release_eval_dir, eval_zip, root_name=eval_name)
    _zip_folder(release_noise_dir.parent, noise_zip, root_name="noise_models")

    application_dir = tmp_path / "application"
    application_dir.mkdir()
    (application_dir / "figure.tif").write_bytes(b"application")
    _zip_folder(application_dir, staging / "application.zip", root_name="application")
    # Model packages are deliberately outside the dataset catalog and may have any ZIP layout.
    with zipfile.ZipFile(staging / "model_a.lisai.zip", "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("artifacts/checkpoint.pth", b"model")

    catalog = build_dataset_catalog(data_root, staging, progress=False)

    assert catalog["schema_version"] == 1
    assert list(catalog["datasets"]) == [eval_name, train_name]
    train_entry = catalog["datasets"][train_name]
    assert train_entry["usage"] == "training"
    assert train_entry["install_path"] == "datasets/training"
    assert train_entry["registry_entry"]["description"] == "training data"
    assert train_entry["dependencies"] == {
        "evaluation_datasets": [eval_name],
        "noise_models": ["monalisa_noise"],
    }
    assert train_entry["archives"][0]["filename"] == train_zip.name
    expected_sha256 = hashlib.sha256(train_zip.read_bytes()).hexdigest()
    assert train_entry["archives"][0]["sha256"] == expected_sha256

    assert catalog["datasets"][eval_name]["dependencies"] == {
        "evaluation_datasets": [],
        "noise_models": [],
    }
    assert catalog["shared"]["noise_models"]["archive"]["filename"] == noise_zip.name
    assert catalog["shared"]["noise_models"]["available"] == ["monalisa_noise"]

    written = json.loads((staging / "dataset_catalog.json").read_text(encoding="utf-8"))
    assert written == catalog
    rendered = json.dumps(catalog)
    assert "application.zip" not in rendered
    assert "model_a.lisai.zip" not in rendered
    # Crucially, the source-only historical evaluation is ignored.
    assert "TestDataset_Gag_low_high" not in rendered


def test_build_dataset_catalog_requires_staged_evaluation_dependency(tmp_path: Path):
    data_root = tmp_path / "lisai"
    staging = tmp_path / "staging"
    staging.mkdir()
    _registry(
        data_root,
        {
            "train": {"usage": "training", "data_format": "single"},
            "eval": {"usage": "evaluation", "data_format": "single"},
        },
    )

    release_train = tmp_path / "release_train" / "train"
    release_train.mkdir(parents=True)
    _write_yaml(
        release_train / "runs" / "model" / "run" / "evaluations" / "eval" / "best" / "evaluation.yaml",
        {"dataset": {"name": "eval", "usage": "evaluation"}},
    )
    _zip_folder(release_train, staging / "train.zip")

    with pytest.raises(ReleaseValidationError, match="has no staged archive"):
        build_dataset_catalog(data_root, staging, progress=False)

    assert not (staging / "dataset_catalog.json").exists()


def test_build_dataset_catalog_requires_referenced_noise_model_in_shared_archive(tmp_path: Path):
    data_root = tmp_path / "lisai"
    staging = tmp_path / "staging"
    staging.mkdir()
    _registry(data_root, {"train": {"usage": "training", "data_format": "single"}})

    release_train = tmp_path / "release_train" / "train"
    release_train.mkdir(parents=True)
    _write_yaml(
        release_train / "runs" / "hdn" / "run" / "config_train.yaml",
        {"noise_model": "needed_noise"},
    )
    _zip_folder(release_train, staging / "train.zip")

    other_noise = tmp_path / "noise_models"
    (other_noise / "other").mkdir(parents=True)
    (other_noise / "other" / "GMM.npz").write_bytes(b"other")
    (other_noise / "other" / "norm_prm.json").write_text("{}", encoding="utf-8")
    _zip_folder(other_noise, staging / "noise_models.zip", root_name="noise_models")

    with pytest.raises(ReleaseValidationError, match="needed_noise"):
        build_dataset_catalog(data_root, staging, progress=False)


def test_build_dataset_catalog_uses_noise_model_archive_as_authoritative_release(tmp_path: Path):
    data_root = tmp_path / "lisai"
    staging = tmp_path / "staging"
    staging.mkdir()
    _registry(data_root, {"train": {"usage": "training", "data_format": "single"}})

    release_train = tmp_path / "release_train" / "train"
    release_train.mkdir(parents=True)
    _write_yaml(
        release_train / "runs" / "hdn" / "run" / "config_train.yaml",
        {"noise_model": "release_noise"},
    )
    _zip_folder(release_train, staging / "train.zip")

    # The staged noise model is sufficient; it need not be an exact copy of a
    # corresponding local source directory.
    release_noise = tmp_path / "noise_models"
    (release_noise / "release_noise").mkdir(parents=True)
    (release_noise / "release_noise" / "GMM.npz").write_bytes(b"gmm")
    (release_noise / "release_noise" / "norm_prm.json").write_text("{}", encoding="utf-8")
    (release_noise / "release_noise" / "README.txt").write_text("extra", encoding="utf-8")
    _zip_folder(release_noise, staging / "noise_models.zip", root_name="noise_models")

    catalog = build_dataset_catalog(data_root, staging, progress=False)
    assert catalog["datasets"]["train"]["dependencies"]["noise_models"] == ["release_noise"]


def test_build_dataset_catalog_rejects_duplicate_files_across_dataset_parts(tmp_path: Path):
    data_root = tmp_path / "lisai"
    staging = tmp_path / "staging"
    staging.mkdir()
    _registry(data_root, {"dataset_a": {"usage": "training", "data_format": "single"}})

    part_a = tmp_path / "part_a" / "dataset_a"
    part_b = tmp_path / "part_b" / "dataset_a"
    part_a.mkdir(parents=True)
    part_b.mkdir(parents=True)
    (part_a / "same.tif").write_bytes(b"a")
    (part_b / "same.tif").write_bytes(b"b")
    _zip_folder(part_a, staging / "dataset_a_01.zip")
    _zip_folder(part_b, staging / "dataset_a_02.zip")

    with pytest.raises(ReleaseValidationError, match="duplicated across its archives"):
        build_dataset_catalog(data_root, staging, progress=False)
