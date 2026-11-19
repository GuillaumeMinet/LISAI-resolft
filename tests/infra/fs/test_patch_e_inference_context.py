"""Patch E: inference and promoted models use an explicitly selected workspace."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.config.io.yaml import save_yaml
from lisai.evaluation.saved_run import SavedTrainingRun, load_saved_run, resolve_run_dir
from lisai.infra.paths import Paths
from lisai.promoted_models.package import load_promoted_model, load_promoted_model_from_dir
from lisai.promoted_models.registry import (
    load_promoted_model_registry,
    register_promoted_model,
)
from lisai.promoted_models.schema import (
    PromotedModelArtifacts,
    PromotedModelDefinition,
    PromotedModelManifest,
    PromotedModelRegistryEntry,
    PromotedModelSource,
    PromotedTrainingData,
)


def _training_config():
    return {
        "experiment": {"mode": "train", "exp_name": "same_model"},
        "routing": {"data_subfolder": "raw"},
        "data": {"dataset_name": "dataset", "paired": False, "input": "inp", "patch_size": 64},
        "model": {"architecture": "unet", "parameters": {}},
        "training": {"n_epochs": 1},
        "saving": {"enabled": True, "state_dict": True, "entire_model": False},
    }


def _manifest():
    return PromotedModelManifest(
        name="shared-name",
        model=PromotedModelDefinition(task="custom", architecture="unet"),
        training_data=PromotedTrainingData(dataset="dataset"),
        source=PromotedModelSource(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV", run_name="same_model",
            run_status="completed", checkpoint_selector="best",
            checkpoint_filename="weights.pt",
        ),
        artifacts=PromotedModelArtifacts(config="config_train.yaml", weights="weights.pt"),
    )


def _install_bundle(paths: Paths) -> Path:
    model_dir = paths.promoted_model_dir(model_name="shared-name")
    model_dir.mkdir(parents=True)
    save_yaml(_training_config(), model_dir / "config_train.yaml")
    (model_dir / "weights.pt").write_bytes(b"dummy weights")
    save_yaml(_manifest().model_dump(mode="json", exclude_none=True), model_dir / "lisai_model.yaml")
    register_promoted_model(
        name="shared-name",
        entry=PromotedModelRegistryEntry(
            source_run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            path="shared-name",
            created_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        ),
        paths=paths,
    )
    return model_dir


def test_promoted_registry_and_bundle_are_independent_a_b_a(settings_factory):
    a, b = Paths(settings_factory("promoted_a")), Paths(settings_factory("promoted_b"))
    a_dir = _install_bundle(a)
    assert load_promoted_model_registry(paths=b).models == {}
    b_dir = _install_bundle(b)

    for paths, expected in ((a, a_dir), (b, b_dir), (a, a_dir)):
        promoted = load_promoted_model("shared-name", paths=paths)
        assert promoted.model_dir == expected
        assert promoted.saved_run.dataset_name == "dataset"
        assert promoted.saved_run.split_manifest is None
        assert set(load_promoted_model_registry(paths=paths).models) == {"shared-name"}
    assert a_dir != b_dir
    assert a.promoted_model_registry_path() != b.promoted_model_registry_path()


def test_ordinary_run_keeps_manifest_and_promoted_copy_omits_it(settings_factory):
    paths = Paths(settings_factory("manifest"))
    run_dir = resolve_run_dir(
        dataset_name="dataset", subfolder="models", exp_name="same_model", paths=paths
    )
    run_dir.mkdir(parents=True)
    save_yaml(_training_config(), paths.cfg_train_path(run_dir=run_dir))
    manifest_path = paths.split_manifest_path(run_dir=run_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"split": "marker", "test_ids": [1, 3]}), encoding="utf-8")

    saved = load_saved_run(run_dir, paths=paths)
    assert saved.split_manifest == {"split": "marker", "test_ids": [1, 3]}
    promoted = load_promoted_model_from_dir(_install_bundle(paths))
    assert promoted.saved_run.split_manifest is None


def test_inference_runtime_receives_explicit_paths_a_b_a(settings_factory, monkeypatch, tmp_path):
    import lisai.evaluation.runtime as module

    a, b = Paths(settings_factory("runtime_a")), Paths(settings_factory("runtime_b"))
    calls = []

    def fake_checkpoint(saved_run, *, paths, **kwargs):
        calls.append(paths)
        checkpoint = tmp_path / "weights.pt"
        checkpoint.write_bytes(b"not loaded")
        return "state_dict", checkpoint

    monkeypatch.setattr(module, "_resolve_checkpoint_path", fake_checkpoint)
    monkeypatch.setattr(module, "_load_state_dict_model", lambda *args, **kwargs: (object(), 1))
    saved = SimpleNamespace(default_tiling_size=128)
    for paths in (a, b, a):
        runtime = module.initialize_runtime(saved_run=saved, paths=paths, device="cpu")
        assert runtime.tiling_size == 128
        assert runtime.resolved_epoch == 1
    assert calls == [a, b, a]


def test_internal_interfaces_reject_missing_paths():
    with pytest.raises(TypeError, match="paths"):
        load_promoted_model_registry()
    with pytest.raises(TypeError, match="paths"):
        load_saved_run(Path("/tmp/example"))


def test_promoted_inference_config_selection_stays_in_same_context(settings_factory):
    from lisai.promoted_models.inference_config import resolve_promoted_inference_config_source
    from lisai.promoted_models.package import set_promoted_model_config

    a, b = Paths(settings_factory("config_a")), Paths(settings_factory("config_b"))
    for paths, samples in ((a, 5), (b, 10)):
        cfg_dir = paths.settings.INFERENCE_CONFIG_DIR
        cfg_dir.mkdir(parents=True, exist_ok=True)
        save_yaml({"apply": {"inference": {"lvae_num_samples": samples}}}, cfg_dir / "experiment.yml")
        _install_bundle(paths)

    for paths, samples in ((a, 5), (b, 10), (a, 5)):
        source = resolve_promoted_inference_config_source("experiment", stg=paths.settings)
        assert source == paths.settings.INFERENCE_CONFIG_DIR / "experiment.yml"
        promoted = set_promoted_model_config("shared-name", "experiment", paths=paths)
        assert f"lvae_num_samples: {samples}" in promoted.inference_config_path.read_text(encoding="utf-8")


def test_promoted_apply_default_output_uses_selected_paths_a_b_a(settings_factory, monkeypatch, tmp_path):
    import importlib
    from lisai.config.models.inference import ApplyDefaults

    app = importlib.import_module("lisai.evaluation.run_apply_model")
    a, b = Paths(settings_factory("apply_a")), Paths(settings_factory("apply_b"))
    _install_bundle(a)
    _install_bundle(b)
    input_dir = tmp_path / "input_images"
    input_dir.mkdir()
    destinations = []
    monkeypatch.setattr(app, "initialize_runtime", lambda **kwargs: SimpleNamespace(tiling_size=64))
    monkeypatch.setattr(app, "resolve_prediction_inputs", lambda path, **kwargs: (path, [], None))
    monkeypatch.setattr(app, "_create_apply_save_folder", lambda path, **kwargs: destinations.append(Path(path)) or Path(path))
    monkeypatch.setattr(app, "_resolve_apply_files_for_output", lambda *args, **kwargs: [])
    cfg = ApplyDefaults.model_validate({"saving": {"mode": "default", "save_input_mode": "never"}})

    expected = []
    for paths in (a, b, a):
        app.run_apply_model(
            cfg, model_dataset="", model_subfolder="promoted", model_name="shared-name",
            data_path=input_dir, promoted_model_name="shared-name", paths=paths,
        )
        expected.append(paths.inference_output_dir(source_name="input_images", model_name="shared-name"))
    assert destinations == expected
    assert destinations[0] != destinations[1]


def test_python_evaluate_uses_explicit_settings(settings_factory, monkeypatch):
    from lisai import api

    a, b = settings_factory("api_a"), settings_factory("api_b")
    captured = []
    monkeypatch.setattr(api, "run_evaluate", lambda cfg, *, paths, **kw: captured.append(paths.settings))
    for settings in (a, b, a):
        api.evaluate(object(), settings=settings)
    assert captured == [a, b, a]


def test_old_prediction_signature_is_no_longer_public(settings_factory):
    from lisai import api
    from lisai.config.models.inference import ApplyDefaults

    settings = settings_factory("norm_api")
    with pytest.raises(TypeError):
        api.predict(ApplyDefaults(), settings=settings, input_normalization={"mode": "image"})
