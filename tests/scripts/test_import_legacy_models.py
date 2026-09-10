from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_script():
    script_path = (
        Path(__file__).resolve().parents[2] / "src" / "scripts" / "import_legacy_models.py"
    )
    spec = importlib.util.spec_from_file_location("import_legacy_models_for_tests", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lvae_translation_keeps_checkpoint_compatible_batch_norm():
    script = _import_script()
    legacy_cfg = {
        "is_lvae": True,
        "model_prm": {
            "num_latents": 2,
            "z_dims": [16, 16],
            "batchnorm": True,
            "img_shape": [64, 64],
            "norm_prm": {"data_mean": 0.0, "data_std": 1.0},
        },
        "data_prm": {
            "subfolder": "old_data",
            "batch_size": 1,
            "patch_size": 64,
        },
        "training_prm": {
            "n_epochs": 5,
            "lr": 0.001,
            "pbar": True,
        },
        "normalization": {},
        "saving_prm": {},
        "noise_model": "Noise0",
    }

    current = script.translate_legacy_config(
        legacy_cfg,
        target_dataset="Fixed_vimentin",
        target_model_subfolder="Denoising/HDN",
        target_run_name="HDN_imported",
    )
    params = current["model"]["parameters"]

    assert current["model"]["architecture"] == "lvae"
    assert current["data"]["dataset_name"] == "Fixed_vimentin"
    assert current["routing"]["models_subfolder"] == "Denoising/HDN"
    assert current["training"]["learning_rate"] == pytest.approx(0.001)
    assert current["training"]["progress_bar"] is True
    assert params["norm"] == "batch"
    assert "img_shape" not in params
    assert "norm_prm" not in params


def test_unet_rcan_translation_normalizes_legacy_upsampling_fields():
    script = _import_script()
    legacy_cfg = {
        "model_architecture": "unetrcan",
        "model_prm": {
            "upsampling_factor": 2,
            "UNet_prm": {
                "feat": 16,
                "depth": 2,
                "in_channels": 1,
                "out_channels": 1,
                "activation": "swish",
                "norm": "group",
                "gr_norm": 8,
            },
            "RCAN_prm": {
                "out_channels": 1,
                "num_features": 16,
                "num_rg": 2,
                "num_rcab": 2,
                "reduction": 4,
                "upsamp": 2,
            },
        },
        "data_prm": {"subfolder": "old_data"},
    }

    section = script._legacy_model_section(legacy_cfg)
    params = section["parameters"]

    assert section["architecture"] == "unet_rcan"
    assert params["upsampling_net"] == "rcan"
    assert params["UNet_prm"]["remove_skip_con"] == 1
    assert params["RCAN_prm"]["upsamp_kernel_factor"] == 2
    assert "upsamp" not in params["RCAN_prm"]


def test_unet_rcan_translation_preserves_explicit_kernel_factor():
    script = _import_script()
    legacy_cfg = {
        "model_architecture": "unetrcan",
        "model_prm": {
            "upsampling_factor": 4,
            "UNet_prm": {
                "feat": 16,
                "depth": 2,
                "in_channels": 1,
                "out_channels": 1,
                "activation": "swish",
                "norm": "group",
                "gr_norm": 8,
            },
            "RCAN_prm": {
                "out_channels": 1,
                "num_features": 16,
                "num_rg": 2,
                "num_rcab": 2,
                "reduction": 4,
                "upsamp": 4,
                "upsamp_kernel_factor": 1,
            },
        },
        "data_prm": {"subfolder": "old_data"},
    }

    section = script._legacy_model_section(legacy_cfg)

    assert section["parameters"]["RCAN_prm"]["upsamp_kernel_factor"] == 1


def test_legacy_denoising_unet_rcan_translation_sets_large_default_tiling_size():
    script = _import_script()
    legacy_cfg = {
        "model_architecture": "unetrcan",
        "model_prm": {
            "upsampling_factor": 1,
            "UNet_prm": {"feat": 16, "depth": 2, "in_channels": 1, "out_channels": 1},
            "RCAN_prm": {"out_channels": 1, "num_features": 16, "num_rg": 2, "num_rcab": 2},
        },
        "data_prm": {"subfolder": "old_data"},
        "training_prm": {},
        "normalization": {},
        "saving_prm": {},
    }

    current = script.translate_legacy_config(
        legacy_cfg,
        target_dataset="Fixed_vimentin",
        target_model_subfolder="denoising_legacy",
        target_run_name="UNetRCAN_single_to_avg",
    )

    assert current["inference"]["default_tiling_size"] == 2000


def test_legacy_upsampling_unet_rcan_translation_uses_architecture_default_tiling():
    script = _import_script()
    legacy_cfg = {
        "model_architecture": "unetrcan",
        "model_prm": {
            "upsampling_factor": 2,
            "UNet_prm": {"feat": 16, "depth": 2, "in_channels": 1, "out_channels": 1},
            "RCAN_prm": {"out_channels": 1, "num_features": 16, "num_rg": 2, "num_rcab": 2},
        },
        "data_prm": {"subfolder": "old_data"},
        "training_prm": {},
        "normalization": {},
        "saving_prm": {},
    }

    current = script.translate_legacy_config(
        legacy_cfg,
        target_dataset="Fixed_vimentin",
        target_model_subfolder="Upsamp",
        target_run_name="UNetRCAN_upsamp",
    )

    assert "inference" not in current


def test_checkpoint_selection_uses_loss_best_epoch_when_direct_best_is_missing(tmp_path: Path):
    script = _import_script()
    (tmp_path / "loss.txt").write_text(
        "epoch train_loss val_loss\n"
        "0 1.0 0.7\n"
        "1 0.8 0.4\n"
        "2 0.9 0.6\n",
        encoding="utf-8",
    )
    (tmp_path / "model_epoch_1_state_dict.pt").write_bytes(b"")
    (tmp_path / "model_last_state_dict.pt").write_bytes(b"")

    best, last = script.select_legacy_checkpoints(tmp_path)

    assert best.selector == "best"
    assert best.source_path.name == "model_epoch_1_state_dict.pt"
    assert best.epoch == 1
    assert last.selector == "last"
    assert last.source_path.name == "model_last_state_dict.pt"


def test_lvae_noise_model_check_requires_canonical_files(tmp_path: Path):
    script = _import_script()

    class FakePaths:
        def noise_model_path(self, *, noiseModel_name: str):
            return tmp_path / f"{noiseModel_name}.pkl"

        def noise_model_norm_prm_path(self, *, noiseModel_name: str):
            return tmp_path / f"{noiseModel_name}_norm.json"

    spec = script.LegacyModelSpec(
        architecture="lvae",
        parameters=object(),
        runtime_img_shape=(64, 64),
        model_norm_prm={},
        noise_model_name="Noise0",
    )

    with pytest.raises(FileNotFoundError, match="Canonical noise model files are missing"):
        script._verify_noise_model_available(spec, FakePaths())

    (tmp_path / "Noise0.pkl").write_bytes(b"")
    (tmp_path / "Noise0_norm.json").write_text("{}", encoding="utf-8")

    script._verify_noise_model_available(spec, FakePaths())


def test_execute_import_keeps_loss_and_log_in_run_root_and_saves_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script = _import_script()
    legacy_run = tmp_path / "legacy_run"
    target_run = tmp_path / "target_run"
    legacy_run.mkdir()
    (legacy_run / "config_train.json").write_text("{}", encoding="utf-8")
    (legacy_run / "loss.txt").write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.0 1.2\n",
        encoding="utf-8",
    )
    (legacy_run / "train_log.log").write_text("legacy log\n", encoding="utf-8")

    monkeypatch.setattr(script, "prune_config_for_saving", lambda _cfg: {"config": True})

    def fake_save_yaml(cfg, path):
        Path(path).write_text("config: true\n", encoding="utf-8")

    monkeypatch.setattr(script, "save_yaml", fake_save_yaml)
    monkeypatch.setattr(script, "_torch_save", lambda _payload, path: Path(path).write_bytes(b"pt"))

    metadata_calls = []
    monkeypatch.setattr(
        script,
        "write_run_metadata_atomic",
        lambda run_dir, metadata: metadata_calls.append((Path(run_dir), metadata)),
    )

    plot_calls = []

    def fake_save_loss_plot_for_run(**kwargs):
        plot_calls.append(kwargs)
        out = Path(kwargs["run_dir"]) / "loss_plot.png"
        out.write_text("fake plot", encoding="utf-8")
        return out

    monkeypatch.setattr(script, "save_loss_plot_for_run", fake_save_loss_plot_for_run)

    plan = SimpleNamespace(
        job=SimpleNamespace(
            legacy_model_path=legacy_run,
            target_dataset="Dataset",
            target_model_subfolder="Models/Subfolder",
            copy_small_legacy_artifacts=True,
        ),
        target_run_dir=target_run,
        resolved_cfg=SimpleNamespace(model=SimpleNamespace(architecture="unet")),
        checkpoints=(),
        metadata={"run": "metadata"},
    )

    script.execute_import(plan)

    assert (target_run / "loss.txt").read_text(encoding="utf-8").startswith("Epoch")
    assert (target_run / "train_log.log").read_text(encoding="utf-8") == "legacy log\n"
    assert not (target_run / "legacy_origin" / "loss.txt").exists()
    assert not (target_run / "legacy_origin" / "train_log.log").exists()
    assert (target_run / "legacy_origin" / "config_train.json").exists()
    assert (target_run / "loss_plot.png").exists()
    assert len(plot_calls) == 1
    assert plot_calls[0]["run_dir"] == target_run
    assert metadata_calls == [(target_run, {"run": "metadata"})]
