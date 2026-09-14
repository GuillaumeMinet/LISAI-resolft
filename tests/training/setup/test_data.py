from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import lisai.training.setup.data as data_mod
from lisai.preprocess.core.dataset_registry import DatasetRegistry


def _write_preprocess_registry_entry(registry_path: Path, *, dataset_name: str, data_format: str) -> None:
    registry = DatasetRegistry(registry_path)
    registry.update_after_preprocess(
        dataset_name=dataset_name,
        data_type="recon",
        data_format=data_format,
        structure=["inp"],
        outputs=[{"key": "inp", "path": "inp", "role": "inp", "axes": "YX"}],
        result=SimpleNamespace(n_files=1, n_frames=None, snr_levels=None, timepoints=None),
        split_summary={"counts": {"train": 1, "val": 0, "test": 0}},
    )
    registry.data[dataset_name]["shape"] = [1, 2, 3]
    registry.save()


def test_prepare_data_returns_prepared_training_data(monkeypatch, tmp_path: Path):
    resolved_kwargs = {}

    def fake_resolved(**kwargs):
        resolved_kwargs.update(kwargs)
        return "resolved_data_cfg"

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="unet"),
        normalization={"norm_prm": {"mean": 1.0}},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    registry_path = tmp_path / "dataset_registry.yml"
    _write_preprocess_registry_entry(registry_path, dataset_name="demo", data_format="single")
    runtime = SimpleNamespace(
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder, usage="training": Path("/tmp/data"),
            dataset_registry_path=lambda: registry_path,
        )
    )

    monkeypatch.setattr(
        data_mod,
        "make_training_loaders",
        lambda config, split_manifest=None, return_split_manifest=False: (
            "train_loader",
            "val_loader",
            {"std": 2.0},
            {"patch": 64},
            split_manifest,
        ),
    )

    prepared = data_mod.prepare_data(cfg, runtime)

    assert isinstance(prepared, data_mod.PreparedTrainingData)
    assert prepared.train_loader == "train_loader"
    assert prepared.val_loader == "val_loader"
    assert prepared.data_norm_prm == {"mean": 1.0}
    assert prepared.model_norm_prm == {"std": 2.0}
    assert prepared.patch_info == {"patch": 64}
    assert prepared.split_manifest is None
    assert resolved_kwargs == {
        "data_dir": Path("/tmp/data"),
        "norm_prm": {"mean": 1.0},
        "dataset_info": {
            "data_format": "single",
            "for_training": True,
            "usage": "training",
            "defaults": {"recon": {"input": "inp", "target": None, "eval_gt": None}},
            "outputs": {"recon": [{"key": "inp", "path": "inp", "role": "inp", "axes": "YX"}]},
            "size": {"recon": {"n_files": 1}},
            "split": {"recon": {"counts": {"train": 1, "val": 0, "test": 0}}},
            "structure": {"recon": ["inp"]},
            "shape": [1, 2, 3],
        },
        "volumetric": False,
    }



def test_prepare_data_uses_noise_model_metadata_for_lvae(monkeypatch):
    resolved_kwargs = {}
    calls = []

    def fake_resolved(**kwargs):
        resolved_kwargs.update(kwargs)
        return "resolved_data_cfg"

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="lvae"),
        normalization={"norm_prm": {"mean": 0.0}, "load_from_noise_model": True},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    runtime = SimpleNamespace(
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder, usage="training": Path("/tmp/data"),
            dataset_registry_path=lambda: Path("/tmp/registry.yaml"),
        )
    )

    def fake_resolve_noise_model_metadata(cfg_arg, paths_arg):
        calls.append((cfg_arg, paths_arg))
        return {"mean": 3.0}

    monkeypatch.setattr(data_mod, "resolve_noise_model_metadata", fake_resolve_noise_model_metadata)
    monkeypatch.setattr(
        data_mod,
        "make_training_loaders",
        lambda config, split_manifest=None, return_split_manifest=False: (
            "train_loader",
            "val_loader",
            {"std": 2.0},
            None,
            split_manifest,
        ),
    )

    prepared = data_mod.prepare_data(cfg, runtime)

    assert calls == [(cfg, runtime.paths)]
    assert prepared.data_norm_prm == {"mean": 3.0}
    assert resolved_kwargs["norm_prm"] == {"mean": 3.0}


def test_prepare_data_reuses_origin_manifest_for_continue(monkeypatch, tmp_path: Path):
    manifest = {"version": 1, "splits": {"train": [], "val": [], "test": []}}
    origin_run_dir = tmp_path / "origin"
    origin_run_dir.mkdir()
    (origin_run_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_resolved(**kwargs):
        return SimpleNamespace(
            prep_before=False,
            split_manifest=SimpleNamespace(retrain_policy="reuse"),
        )

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="unet"),
        experiment=SimpleNamespace(mode="continue_training", origin_run_dir=str(origin_run_dir)),
        normalization={"norm_prm": None},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    runtime = SimpleNamespace(
        run_dir=origin_run_dir,
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder, usage="training": tmp_path / "data",
            dataset_registry_path=lambda: tmp_path / "registry.yaml",
            split_manifest_path=lambda run_dir: Path(run_dir) / "split_manifest.json",
        ),
    )
    captured = {}

    def fake_make_training_loaders(config, split_manifest=None, return_split_manifest=False):
        captured["split_manifest"] = split_manifest
        return "train_loader", "val_loader", None, None, split_manifest

    monkeypatch.setattr(data_mod, "make_training_loaders", fake_make_training_loaders)

    prepared = data_mod.prepare_data(cfg, runtime)

    assert captured["split_manifest"] == manifest
    assert prepared.split_manifest == manifest


def test_prepare_data_errors_when_retrain_reuse_manifest_is_missing(monkeypatch, tmp_path: Path):
    origin_run_dir = tmp_path / "origin"
    origin_run_dir.mkdir()

    def fake_resolved(**kwargs):
        return SimpleNamespace(
            prep_before=False,
            split_manifest=SimpleNamespace(retrain_policy="reuse"),
        )

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="unet"),
        experiment=SimpleNamespace(mode="retrain", origin_run_dir=str(origin_run_dir)),
        normalization={"norm_prm": None},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    runtime = SimpleNamespace(
        run_dir=tmp_path / "new_run",
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder, usage="training": tmp_path / "data",
            dataset_registry_path=lambda: tmp_path / "registry.yaml",
            split_manifest_path=lambda run_dir: Path(run_dir) / "split_manifest.json",
        ),
    )

    try:
        data_mod.prepare_data(cfg, runtime)
    except FileNotFoundError as exc:
        assert "Missing split manifest" in str(exc)
    else:
        raise AssertionError("Expected missing retrain reuse manifest to fail.")


def test_prepare_data_creates_new_manifest_for_retrain_new_policy(monkeypatch, tmp_path: Path):
    def fake_resolved(**kwargs):
        return SimpleNamespace(
            prep_before=False,
            split_manifest=SimpleNamespace(retrain_policy="new"),
        )

    cfg = SimpleNamespace(
        model=SimpleNamespace(architecture="unet"),
        experiment=SimpleNamespace(mode="retrain", origin_run_dir=str(tmp_path / "origin")),
        normalization={"norm_prm": None},
        data=SimpleNamespace(dataset_name="demo", resolved=fake_resolved),
        routing=SimpleNamespace(data_subfolder="raw"),
    )
    runtime = SimpleNamespace(
        run_dir=tmp_path / "new_run",
        paths=SimpleNamespace(
            dataset_dir=lambda dataset_name, data_subfolder, usage="training": tmp_path / "data",
            dataset_registry_path=lambda: tmp_path / "registry.yaml",
        ),
    )
    captured = {}
    created_manifest = {"version": 1, "splits": {"train": [], "val": [], "test": []}}

    def fake_make_training_loaders(config, split_manifest=None, return_split_manifest=False):
        captured["split_manifest"] = split_manifest
        return "train_loader", "val_loader", None, None, created_manifest

    monkeypatch.setattr(data_mod, "make_training_loaders", fake_make_training_loaders)

    prepared = data_mod.prepare_data(cfg, runtime)

    assert captured["split_manifest"] is None
    assert prepared.split_manifest == created_manifest
