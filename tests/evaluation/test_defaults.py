from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.evaluation.defaults as defaults_mod
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.yaml import save_yaml
from lisai.evaluation.defaults import (
    resolve_apply_options,
    resolve_apply_save_input,
    resolve_evaluate_options,
    resolve_inference_config_path,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def inference_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs" / "inference"
    fake_settings = SimpleNamespace(
        INFERENCE_CONFIG_DIR=config_dir,
        INFERENCE_DEFAULT_CONFIG_NAME="local/defaults",
        CONFIG_SUFFIXES=(".yml", ".yaml"),
    )
    monkeypatch.setattr(
        defaults_mod,
        "inference_config_paths",
        ConfigPathResolver("inference", stg=fake_settings),
    )
    return config_dir


def _write_local_defaults(config_dir: Path, content: str) -> Path:
    path = config_dir / "local" / "defaults.yml"
    _write(path, content)
    return path


def _complete_apply_config(*, tiling_size=512, crop_size=128, saving=None) -> dict:
    apply = {
        "checkpoint": {"epoch_number": None, "best_or_last": "best"},
        "input": {
            "filters": ["tiff", "tif"],
            "skip_if_contain": None,
            "stack_selection_idx": None,
            "limit_n_imgs": None,
            "timelapse_max": None,
        },
        "inference": {
            "crop_size": crop_size,
            "keep_original_shape": True,
            "tiling_size": tiling_size,
            "lvae_num_samples": 20,
            "downsamp": None,
            "fill_factor": None,
            "dark_frame_context_length": False,
        },
        "postprocess": {
            "denormalize": True,
            "color_code": {
                "enabled": False,
                "colormap": "turbo",
                "saturation": 0.35,
                "add_colorbar": True,
                "zstep": 0.4,
            },
        },
    }
    if saving is not None:
        apply["saving"] = saving
    return {"apply": apply}


def _complete_evaluate_config(*, tiling_size="off") -> dict:
    return {
        "evaluate": {
            "checkpoint": {"epoch_number": None, "best_or_last": "best"},
            "data": {
                "split": "test",
                "eval_gt": None,
                "overrides": None,
                "limit_n_imgs": None,
                "timelapse_max": None,
            },
            "inference": {
                "tiling_size": tiling_size,
                "crop_size": None,
                "lvae_num_samples": 20,
                "ch_out": 1,
            },
            "metrics": None,
        }
    }


def test_resolve_inference_config_path_defaults_to_local_defaults(inference_config_dir: Path):
    defaults_path = _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: 256\n")

    resolved = resolve_inference_config_path(None)

    assert resolved == defaults_path.resolve()


def test_named_lookup_prefers_local_config(inference_config_dir: Path):
    local_path = inference_config_dir / "local" / "fast.yml"
    tracked_path = inference_config_dir / "fast.yml"
    _write(local_path, "apply:\n  inference:\n    tiling_size: 256\n")
    save_yaml(_complete_apply_config(tiling_size=1024), tracked_path)

    assert resolve_inference_config_path("fast") == local_path.resolve()


def test_local_named_config_inherits_local_defaults_then_cli(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        """
apply:
  inference:
    tiling_size: 256
    fill_factor: 0.5
  postprocess:
    denormalize: false
    color_code:
      colormap: turbo
      saturation: 0.35
""".strip() + "\n",
    )
    _write(
        inference_config_dir / "local" / "fast_upsamp.yml",
        """
apply:
  inference:
    tiling_size: 512
    crop_size: 128
    fill_factor: 0.75
""".strip() + "\n",
    )

    resolved = resolve_apply_options(config="fast_upsamp", crop_size=200)

    assert resolved["tiling_size"] == 512
    assert resolved["crop_size"] == 200
    assert resolved["fill_factor"] == pytest.approx(0.75)
    assert resolved["denormalize_output"] is False
    assert resolved["color_code_prm"]["colormap"] == "turbo"


def test_local_defaults_accept_legacy_flat_layout(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        """
apply:
  downsamp: 2
  fill_factor: null
""".strip() + "\n",
    )

    resolved = resolve_apply_options()

    assert resolved["downsamp"] == 2
    assert resolved["fill_factor"] is None


def test_standalone_config_does_not_inherit_local_defaults(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    tiling_size: 256\n  postprocess:\n    denormalize: false\n",
    )
    standalone = _complete_apply_config(tiling_size=1024)
    save_yaml(standalone, inference_config_dir / "portable.yml")

    resolved = resolve_apply_options(config="portable")

    assert resolved["tiling_size"] == 1024
    assert resolved["denormalize_output"] is True


def test_standalone_config_must_be_complete(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: 256\n")
    _write(inference_config_dir / "partial.yml", "apply:\n  inference:\n    tiling_size: 512\n")

    with pytest.raises(ValueError, match="Standalone inference config.*incomplete"):
        resolve_apply_options(config="partial")


def test_standalone_evaluate_preserves_tiling_policy_and_cli_override(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  inference:\n    tiling_size: auto\n")
    save_yaml(_complete_evaluate_config(tiling_size="off"), inference_config_dir / "no_tiling.yml")

    resolved = resolve_evaluate_options(config="no_tiling")
    forced = resolve_evaluate_options(config="no_tiling", tiling_size=512)

    assert resolved["tiling_size"] == "off"
    assert forced["tiling_size"] == 512


def test_resolve_evaluate_options_requires_requested_section_in_local_config(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  data:\n    split: test\n")
    _write(inference_config_dir / "local" / "apply_only.yml", "apply:\n  inference:\n    tiling_size: 512\n")

    with pytest.raises(ValueError, match="does not define a 'evaluate' section"):
        resolve_evaluate_options(config="apply_only")


def test_inference_config_rejects_unknown_evaluate_keys(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  data:\n    split: test\n")
    _write(inference_config_dir / "local" / "invalid.yml", "evaluate:\n  test_loader: bad\n")

    with pytest.raises(Exception, match="test_loader"):
        resolve_evaluate_options(config="invalid")


def test_resolve_apply_save_input_uses_local_config_policy(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: auto\n")
    in_place = defaults_mod.ApplyOutputPolicy(mode="in_place")
    default = defaults_mod.ApplyOutputPolicy(mode="default")
    local = SimpleNamespace(INFERENCE_SAVE_INPUT_MODE="if_not_in_place")

    assert resolve_apply_save_input(output_policy=in_place, stg=local) is False
    assert resolve_apply_save_input(output_policy=default, stg=local) is True


def test_local_named_saving_override_overrides_local_config_policy(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: auto\n")
    _write(
        inference_config_dir / "local" / "no_input.yml",
        "apply:\n  saving:\n    save_input_mode: never\n",
    )
    policy = defaults_mod.ApplyOutputPolicy(mode="default")
    local = SimpleNamespace(INFERENCE_SAVE_INPUT_MODE="always")

    assert resolve_apply_save_input(output_policy=policy, config="no_input", stg=local) is False


def test_cli_save_input_override_has_priority(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: auto\n")
    policy = defaults_mod.ApplyOutputPolicy(mode="in_place")
    local = SimpleNamespace(INFERENCE_SAVE_INPUT_MODE="never")

    assert resolve_apply_save_input(output_policy=policy, save_input=True, stg=local) is True
    assert resolve_apply_save_input(output_policy=policy, save_input=False, stg=local) is False


def test_apply_saving_config_overrides_local_config_route(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_inside\n")
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="folder_outside")

    policy = defaults_mod.resolve_apply_output_policy(stg=local)

    assert policy.mode == "folder_inside"


def test_named_local_saving_override_has_priority_over_local_defaults(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_inside\n")
    _write(inference_config_dir / "local" / "special.yml", "apply:\n  saving:\n    mode: in_place\n")
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="folder_outside")

    policy = defaults_mod.resolve_apply_output_policy(config="special", stg=local)

    assert policy.mode == "in_place"


def test_standalone_saving_can_override_local_config_without_local_defaults(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_inside\n")
    cfg = _complete_apply_config(saving={"mode": "folder_outside"})
    save_yaml(cfg, inference_config_dir / "portable.yml")
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(config="portable", stg=local)

    assert policy.mode == "folder_outside"


def test_cli_output_override_has_highest_priority(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_outside\n")
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(
        output_mode="folder_inside",
        stg=local,
    )

    assert policy.mode == "folder_inside"


def test_builtin_post_training_preset_is_used_if_local_file_is_missing(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  inference:\n    tiling_size: 256\n")

    resolved = resolve_evaluate_options(config="post_training")

    assert resolved["tiling_size"] == 256
    assert resolved["best_or_last"] == "both"
    assert resolved["metrics_list"] == ["psnr", "ssim"]
    assert resolved["overwrite"] is True


def test_local_post_training_overrides_builtin_and_inherits_local_defaults(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  inference:\n    tiling_size: 384\n")
    _write(
        inference_config_dir / "local" / "post_training.yml",
        "evaluate:\n  metrics:\n    - custom_metric\n",
    )

    resolved = resolve_evaluate_options(config="post_training")

    assert resolved["tiling_size"] == 384
    assert resolved["best_or_last"] == "both"
    assert resolved["metrics_list"] == ["custom_metric"]
    assert resolved["overwrite"] is True
