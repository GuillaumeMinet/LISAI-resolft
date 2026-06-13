from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import lisai.config.io.resolver as resolver_mod
from lisai.config import save_yaml, settings
from lisai.config.io.resolver import prune_config_for_saving, resolve_config, resolve_config_dict
from lisai.config.models import ResolvedExperiment
from lisai.infra.paths import Paths
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.schema import RunMetadata


def _failed_run_metadata_payload(run_dir: Path, *, checkpoint_name: str | None = None) -> dict:
    return {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAA",
        "run_name": "origin_run",
        "run_index": 0,
        "dataset": "origin_ds",
        "model_subfolder": "HDN",
        "status": "failed",
        "closed_cleanly": True,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": "2026-03-20T10:15:00Z",
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 2,
        "max_epoch": 10,
        "best_val_loss": 0.5,
        "path": f"datasets/origin_ds/models/HDN/{run_dir.name}",
        "group_path": None,
        "recovery_checkpoint_filename": checkpoint_name,
    }


def _registry_entry(
    *,
    input_name: str | None = "",
    target_name: str | None = None,
    usage: str = "training",
    data_format: str = "single",
    input_axes: str = "YX",
    input_data_format_override: str | None = None,
) -> dict:
    outputs = []
    structure = []
    if input_name is not None:
        structure.append(input_name)
        input_output = {
            "key": input_name or "image",
            "path": input_name,
            "role": "inp",
            "axes": input_axes,
        }
        if input_data_format_override is not None:
            input_output["data_format_override"] = input_data_format_override
        outputs.append(input_output)
    if target_name is not None:
        structure.append(target_name)
        outputs.append({"key": target_name, "path": target_name, "role": "gt", "axes": "YX"})

    return {
        "data_format": data_format,
        "usage": usage,
        "for_training": usage == "training",
        "structure": {"recon": structure},
        "outputs": {"recon": outputs},
        "defaults": {
            "recon": {
                "input": input_name,
                "target": target_name,
                "eval_gt": target_name,
            }
        },
    }


@pytest.fixture(autouse=True)
def _fake_dataset_registry(monkeypatch: pytest.MonkeyPatch):
    names = {
        "ds",
        "ds_metadata",
        "ds_template",
        "ds_train",
        "origin_ds",
        "target_ds",
    }
    registry = {name: _registry_entry() for name in names}
    monkeypatch.setattr(resolver_mod, "load_dataset_registry", lambda path: registry)


def test_resolve_config_train_mode_forbids_load_model_section(tmp_path: Path):
    exp_cfg = tmp_path / "exp_train.yml"
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "exp_train"},
            "data": {"dataset_name": "ds_train"},
            "model": {"architecture": "unet", "parameters": {}},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(tmp_path / "ignored_origin"),
                "load_method": "state_dict",
            },
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="load_model"):
        resolve_config(experiment_cfg_path=exp_cfg)


def test_resolve_config_dict_strips_catalog_metadata():
    cfg = resolve_config_dict(
        {
            "metadata": {
                "kind": "example",
                "name": "metadata_smoke",
                "description": "Catalog metadata is not part of the training schema.",
            },
            "experiment": {"mode": "train", "exp_name": "metadata_smoke"},
            "data": {"dataset_name": "ds_metadata"},
            "model": {"architecture": "unet", "parameters": {}},
        }
    )

    assert cfg.experiment.exp_name == "metadata_smoke"
    assert cfg.data.dataset_name == "ds_metadata"


def test_resolve_config_dict_refuses_template_metadata():
    with pytest.raises(ValueError, match="templates must be instantiated"):
        resolve_config_dict(
            {
                "metadata": {"kind": "template", "name": "base"},
                "experiment": {"mode": "train", "exp_name": "template"},
                "data": {"dataset_name": "ds_template"},
            }
        )


def test_resolve_config_dict_refuses_changeme_placeholders():
    with pytest.raises(ValueError, match="data.dataset_name"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "exp"},
                "data": {"dataset_name": "CHANGEME"},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_fills_single_root_registry_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {"demo": _registry_entry(input_name="")},
    )

    cfg = resolve_config_dict(
        {
            "experiment": {"mode": "train", "exp_name": "root_registry_input"},
            "routing": {"data_subfolder": "preprocess/recon"},
            "data": {"dataset_name": "demo"},
            "model": {"architecture": "unet", "parameters": {}},
        }
    )

    assert cfg.data.data_format == "single"
    assert cfg.data.input == ""
    assert cfg.data.registry_data_type == "recon"


def test_resolve_config_dict_requires_explicit_non_root_registry_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {"demo": _registry_entry(input_name="inp", target_name="gt")},
    )

    with pytest.raises(ValidationError, match="registry default is 'inp'"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "explicit_input"},
                "routing": {"data_subfolder": "preprocess/recon"},
                "data": {"dataset_name": "demo", "paired": True},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_requires_explicit_paired_registry_target(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {"demo": _registry_entry(input_name="inp", target_name="gt")},
    )

    with pytest.raises(ValidationError, match="registry default is 'gt'"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "explicit_target"},
                "routing": {"data_subfolder": "preprocess/recon"},
                "data": {"dataset_name": "demo", "paired": True, "input": "inp"},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_uses_format_from_output_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {
            "demo": _registry_entry(
                input_name="inp_single",
                data_format="mltpl_snr",
                input_axes="YX",
                input_data_format_override="single",
            )
        },
    )

    cfg = resolve_config_dict(
        {
            "experiment": {"mode": "train", "exp_name": "single_axes_default"},
            "routing": {"data_subfolder": "preprocess/recon"},
            "data": {"dataset_name": "demo", "input": "inp_single"},
            "model": {"architecture": "unet", "parameters": {}},
        }
    )

    assert cfg.data.input == "inp_single"
    assert cfg.data.data_format == "single"


def test_resolve_config_dict_allows_format_from_output_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {
            "demo": _registry_entry(
                input_name="inp_single",
                data_format="mltpl_snr",
                input_axes="YX",
                input_data_format_override="single",
            )
        },
    )

    cfg = resolve_config_dict(
        {
            "experiment": {"mode": "train", "exp_name": "single_axes_explicit"},
            "routing": {"data_subfolder": "preprocess/recon"},
            "data": {"dataset_name": "demo", "data_format": "single", "input": "inp_single"},
            "model": {"architecture": "unet", "parameters": {}},
        }
    )

    assert cfg.data.data_format == "single"


def test_resolve_config_dict_rejects_single_format_without_output_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {
            "demo": _registry_entry(
                input_name="inp_single",
                data_format="mltpl_snr",
                input_axes="YX",
            )
        },
    )

    with pytest.raises(ValidationError, match="data.data_format"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "single_axes_without_override"},
                "routing": {"data_subfolder": "preprocess/recon"},
                "data": {
                    "dataset_name": "demo",
                    "data_format": "single",
                    "input": "inp_single",
                },
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_rejects_unregistered_prepared_dataset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(resolver_mod, "load_dataset_registry", lambda path: {})

    with pytest.raises(ValidationError, match="not registered"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "missing_registry"},
                "data": {"dataset_name": "missing_ds", "prep_before": True},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_allows_unregistered_unprepared_dataset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(resolver_mod, "load_dataset_registry", lambda path: {})

    cfg = resolve_config_dict(
        {
            "experiment": {"mode": "train", "exp_name": "unprepared"},
            "data": {"dataset_name": "missing_ds", "prep_before": False},
            "model": {"architecture": "unet", "parameters": {}},
        }
    )

    assert cfg.data.dataset_info is None
    assert cfg.data.prep_before is False


def test_resolve_config_dict_rejects_registry_data_format_conflict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {"demo": _registry_entry(input_name="inp")},
    )

    with pytest.raises(ValidationError, match="data.data_format"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "format_conflict"},
                "routing": {"data_subfolder": "preprocess/recon"},
                "data": {"dataset_name": "demo", "data_format": "timelapse", "input": "inp"},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_rejects_evaluation_only_registry_dataset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {"eval_ds": _registry_entry(input_name="inp", usage="evaluation")},
    )

    with pytest.raises(ValidationError, match="evaluation-only"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "eval_only"},
                "routing": {"data_subfolder": "preprocess/recon"},
                "data": {"dataset_name": "eval_ds"},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )


def test_resolve_config_dict_requires_input_when_registry_default_is_ambiguous(monkeypatch: pytest.MonkeyPatch):
    entry = _registry_entry(input_name=None)
    entry["structure"]["recon"] = ["inp_a", "inp_b"]
    entry["outputs"]["recon"] = [
        {"key": "inp_a", "path": "inp_a", "role": "inp", "axes": "YX"},
        {"key": "inp_b", "path": "inp_b", "role": "inp", "axes": "YX"},
    ]
    entry["defaults"]["recon"]["input"] = None
    monkeypatch.setattr(resolver_mod, "load_dataset_registry", lambda path: {"ambiguous": entry})

    with pytest.raises(ValidationError, match="data.input"):
        resolve_config_dict(
            {
                "experiment": {"mode": "train", "exp_name": "ambiguous_input"},
                "routing": {"data_subfolder": "preprocess/recon"},
                "data": {"dataset_name": "ambiguous"},
                "model": {"architecture": "unet", "parameters": {}},
            }
        )



def test_apply_mode_resolution_continue_training_loads_origin_and_applies_sparse_overrides(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {"feat": 8}},
            "training": {"n_epochs": 50, "batch_size": 2},
        },
        origin_cfg_path,
    )

    user_cfg = {
        "experiment": {
            "mode": "continue_training",
            "post_training_inference": False,
            "origin_run_dir": str(origin_run_dir),
        },
        "training": {"n_epochs": 3},
        "load_model": {"checkpoint": {"method": "state_dict", "selector": "last"}},
    }

    merged = resolver_mod._apply_mode_resolution(
        user_cfg=user_cfg,
        mode="continue_training",
        paths=Paths(settings),
    )

    assert resolver_mod._dget(merged, "experiment.mode") == "continue_training"
    assert resolver_mod._dget(merged, "experiment.exp_name") == "origin_exp"
    assert resolver_mod._dget(merged, "experiment.post_training_inference") is False
    assert Path(resolver_mod._dget(merged, "experiment.origin_run_dir")).resolve() == origin_run_dir.resolve()

    assert resolver_mod._dget(merged, "model.architecture") == "unet"
    assert resolver_mod._dget(merged, "model.parameters.feat") == 8

    assert resolver_mod._dget(merged, "training.n_epochs") == 3
    assert resolver_mod._dget(merged, "training.batch_size") == 2
    assert resolver_mod._dget(merged, "load_model.checkpoint.method") == "state_dict"
    assert resolver_mod._dget(merged, "load_model.checkpoint.selector") == "last"



def test_resolve_config_continue_training_requires_load_model_section(tmp_path: Path):
    exp_cfg = tmp_path / "exp_invalid_continue.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training"},
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="load_model"):
        resolve_config(experiment_cfg_path=exp_cfg)



def test_resolve_config_continue_training_forbids_exp_name_override(tmp_path: Path):
    exp_cfg = tmp_path / "exp_continue_with_name.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training", "exp_name": "new_name"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(tmp_path / "origin_run"),
                "load_method": "state_dict",
            },
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="exp_name"):
        resolve_config(experiment_cfg_path=exp_cfg)



def test_resolve_config_continue_training_forbids_loss_function_override(tmp_path: Path):
    exp_cfg = tmp_path / "exp_continue_with_loss.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(tmp_path / "origin_run"),
                "load_method": "state_dict",
            },
            "loss_function": {"name": "CharEdge_loss", "CharEdge_loss_prm": {"alpha": 0.1}},
        },
        exp_cfg,
    )

    with pytest.raises(ValidationError, match="loss_function"):
        resolve_config(experiment_cfg_path=exp_cfg)



def test_resolve_config_retrain_allows_data_and_normalization_overrides(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "routing": {"models_subfolder": "origin_subfolder"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {"feat": 8}},
            "normalization": {"load_from_noise_model": True},
            "noise_model": {"name": "origin_noise"},
            "training": {"n_epochs": 50},
        },
        origin_cfg_path,
    )

    exp_cfg = tmp_path / "retrain.yml"
    save_yaml(
        {
            "experiment": {"mode": "retrain", "exp_name": "retrain_exp"},
            "routing": {"models_subfolder": "retrain_subfolder"},
            "data": {"dataset_name": "target_ds", "patch_size": 32},
            "normalization": {"load_from_noise_model": False},
            "noise_model": {"name": "target_noise"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(origin_run_dir),
                "load_method": "state_dict",
            },
        },
        exp_cfg,
    )

    cfg = resolve_config(experiment_cfg_path=exp_cfg)

    assert cfg.experiment.mode == "retrain"
    assert cfg.experiment.exp_name == "retrain_exp"
    assert Path(cfg.experiment.origin_run_dir).resolve() == origin_run_dir.resolve()
    assert cfg.routing.models_subfolder == "retrain_subfolder"
    assert cfg.data.dataset_name == "target_ds"
    assert cfg.data.patch_size == 32
    assert cfg.model.architecture == "unet"
    assert cfg.normalization.load_from_noise_model is False
    assert cfg.noise_model.name == "target_noise"
    assert cfg.load_model.enabled is True
    assert Path(cfg.load_model.run_dir).resolve() == origin_run_dir.resolve()



def test_prune_config_for_saving_drops_train_only_and_disabled_sections():
    cfg = ResolvedExperiment.model_validate(
        {
            "experiment": {
                "mode": "train",
                "exp_name": "exp_prune",
                "origin_run_dir": "C:/tmp/origin",
            },
            "data": {"dataset_name": "ds"},
            "model": {"architecture": "unet", "parameters": {}},
            "saving": {"enabled": False},
            "tensorboard": {"enabled": False},
            "load_model": {
                "enabled": True,
                "source": "path",
                "run_dir": "C:/tmp/origin",
                "checkpoint": {"method": "state_dict"},
            },
        }
    )

    out = prune_config_for_saving(cfg)

    assert "load_model" not in out
    assert "origin_run_dir" not in out["experiment"]
    assert "saving" not in out
    assert "tensorboard" not in out


def test_resolve_config_dict_matches_file_based_resolution(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "routing": {"models_subfolder": "Upsamp"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {"feat": 8}},
            "training": {"n_epochs": 50},
        },
        origin_cfg_path,
    )

    continue_cfg = {
        "experiment": {"mode": "continue_training"},
        "load_model": {
            "canonical_load": False,
            "model_full_path": str(origin_run_dir),
            "load_method": "state_dict",
            "best_or_last": "last",
        },
    }
    exp_cfg = tmp_path / "continue.yml"
    save_yaml(continue_cfg, exp_cfg)

    resolved_from_dict = resolve_config_dict(continue_cfg)
    resolved_from_file = resolve_config(experiment_cfg_path=exp_cfg)

    assert resolved_from_dict.model_dump() == resolved_from_file.model_dump()


def test_resolve_config_continue_training_auto_selects_safe_checkpoint_and_forces_state_dict(tmp_path: Path):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    paths = Paths(settings)
    origin_cfg_path = paths.cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {}},
            "training": {"n_epochs": 50, "batch_size": 2},
        },
        origin_cfg_path,
    )

    checkpoint_name = "safe_on_divergence.pth"
    checkpoint_path = paths.checkpoint_path(run_dir=origin_run_dir, model_name=checkpoint_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"safe")
    write_run_metadata_atomic(
        origin_run_dir,
        RunMetadata.model_validate(
            _failed_run_metadata_payload(origin_run_dir, checkpoint_name=checkpoint_name),
        ),
    )

    exp_cfg = tmp_path / "continue_safe.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(origin_run_dir),
                "load_method": "full_model",
                "best_or_last": "last",
            },
        },
        exp_cfg,
    )

    cfg = resolve_config(experiment_cfg_path=exp_cfg)

    assert cfg.load_model.checkpoint.filename == checkpoint_name
    assert cfg.load_model.checkpoint.method == "state_dict"
    assert cfg.load_model.checkpoint.selector is None
    assert cfg.load_model.checkpoint.epoch is None


def test_resolve_config_continue_training_uses_project_recovery_defaults_when_origin_has_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    origin_run_dir = tmp_path / "origin_run"
    origin_run_dir.mkdir(parents=True, exist_ok=True)

    origin_cfg_path = Paths(settings).cfg_train_path(run_dir=origin_run_dir)
    save_yaml(
        {
            "experiment": {"mode": "train", "exp_name": "origin_exp"},
            "data": {"dataset_name": "origin_ds", "patch_size": 64},
            "model": {"architecture": "unet", "parameters": {}},
            "training": {"n_epochs": 50, "batch_size": 2},
        },
        origin_cfg_path,
    )

    safe_resume_defaults = settings.project_cfg.recovery.hdn_safe_resume
    monkeypatch.setattr(safe_resume_defaults, "lr_scale", 0.37)
    monkeypatch.setattr(safe_resume_defaults, "force_grad_clip_max_norm", 1.5)

    exp_cfg = tmp_path / "continue_defaults.yml"
    save_yaml(
        {
            "experiment": {"mode": "continue_training"},
            "load_model": {
                "canonical_load": False,
                "model_full_path": str(origin_run_dir),
                "load_method": "state_dict",
                "best_or_last": "last",
            },
        },
        exp_cfg,
    )

    cfg = resolve_config(experiment_cfg_path=exp_cfg)

    safe_resume = cfg.recovery.hdn_safe_resume
    assert safe_resume.lr_scale == pytest.approx(0.37)
    assert safe_resume.force_grad_clip_max_norm == pytest.approx(1.5)
