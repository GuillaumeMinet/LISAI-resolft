from __future__ import annotations

from pathlib import Path
import warnings
from types import SimpleNamespace

import pytest

import lisai.evaluation.defaults as defaults_mod
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.yaml import save_yaml
from lisai.config.models.inference import (
    ApplyDefaults,
    ApplyInferenceOverrides,
    ApplyOverrides,
    EvaluateDefaults,
    EvaluateInferenceOverrides,
    EvaluateOverrides,
)
from lisai.evaluation.defaults import (
    resolve_apply_config,
    resolve_apply_save_input,
    resolve_evaluate_config,
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

    resolved = resolve_apply_config(
        config="fast_upsamp",
        overrides=ApplyOverrides(
            inference=ApplyInferenceOverrides(crop_size=200),
        ),
    )

    assert resolved.inference.tiling_size == 512
    assert resolved.inference.crop_size == 200
    assert resolved.inference.fill_factor == pytest.approx(0.75)
    assert resolved.postprocess.denormalize is False
    assert resolved.postprocess.color_code.colormap == "turbo"


def test_resolve_apply_config_returns_typed_nested_config(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    tiling_size: 256\n    lvae_num_samples: 30\n",
    )
    _write(
        inference_config_dir / "examples" / "hdn_sup.yml",
        "apply:\n  inference:\n    lvae_num_samples: 10\n",
    )

    resolved = resolve_apply_config(
        config="examples/hdn_sup",
        overrides=ApplyOverrides(
            inference=ApplyInferenceOverrides(tiling_size=512),
        ),
    )

    assert isinstance(resolved, ApplyDefaults)
    assert resolved.inference.tiling_size == 512
    assert resolved.inference.lvae_num_samples == 10
    assert resolved.postprocess.denormalize is True


def test_resolve_evaluate_config_returns_typed_nested_config(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        "evaluate:\n  inference:\n    tiling_size: 256\n",
    )

    resolved = resolve_evaluate_config(
        config="post_training",
        overrides=EvaluateOverrides(
            inference=EvaluateInferenceOverrides(tiling_size="off"),
        ),
    )

    assert isinstance(resolved, EvaluateDefaults)
    assert resolved.inference.tiling_size == "off"
    assert resolved.checkpoint.best_or_last == "both"
    assert resolved.metrics == ["psnr", "ssim"]
    assert resolved.saving.overwrite is True


def test_local_defaults_accept_legacy_flat_layout(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        """
apply:
  downsamp: 2
  fill_factor: null
""".strip() + "\n",
    )

    resolved = resolve_apply_config()

    assert resolved.inference.downsamp == 2
    assert resolved.inference.fill_factor is None


def test_named_nonlocal_config_inherits_local_defaults_without_being_overridden(
    inference_config_dir: Path,
):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    tiling_size: 256\n  postprocess:\n    denormalize: false\n",
    )
    _write(
        inference_config_dir / "examples" / "portable.yml",
        "apply:\n  inference:\n    tiling_size: 1024\n",
    )

    resolved = resolve_apply_config(config="examples/portable")

    assert resolved.inference.tiling_size == 1024
    assert resolved.postprocess.denormalize is False


def test_named_nonlocal_config_can_be_sparse(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    tiling_size: 256\n    lvae_num_samples: 30\n",
    )
    _write(
        inference_config_dir / "examples" / "hdn_sup.yml",
        "apply:\n  inference:\n    lvae_num_samples: 10\n",
    )

    resolved = resolve_apply_config(config="examples/hdn_sup")

    assert resolved.inference.lvae_num_samples == 10
    assert resolved.inference.tiling_size == 256


def test_named_config_explicit_null_overrides_local_default(inference_config_dir: Path):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    lvae_num_samples: 30\n",
    )
    _write(
        inference_config_dir / "examples" / "no_lvae_sampling.yml",
        "apply:\n  inference:\n    lvae_num_samples: null\n",
    )

    resolved = resolve_apply_config(config="examples/no_lvae_sampling")

    assert resolved.inference.lvae_num_samples is None


def test_named_nonlocal_evaluate_preserves_tiling_policy_and_cli_override(
    inference_config_dir: Path,
):
    _write_local_defaults(inference_config_dir, "evaluate:\n  inference:\n    tiling_size: auto\n")
    _write(
        inference_config_dir / "examples" / "no_tiling.yml",
        "evaluate:\n  inference:\n    tiling_size: off\n",
    )

    resolved = resolve_evaluate_config(config="examples/no_tiling")
    forced = resolve_evaluate_config(
        config="examples/no_tiling",
        overrides=EvaluateOverrides(
            inference=EvaluateInferenceOverrides(tiling_size=512),
        ),
    )

    assert resolved.inference.tiling_size == "off"
    assert forced.inference.tiling_size == 512


def test_resolve_evaluate_config_requires_requested_section_in_local_config(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  data:\n    split: test\n")
    _write(inference_config_dir / "local" / "apply_only.yml", "apply:\n  inference:\n    tiling_size: 512\n")

    with pytest.raises(ValueError, match="does not define a 'evaluate' section"):
        resolve_evaluate_config(config="apply_only")


def test_inference_config_rejects_unknown_evaluate_keys(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  data:\n    split: test\n")
    _write(inference_config_dir / "local" / "invalid.yml", "evaluate:\n  test_loader: bad\n")

    with pytest.raises(Exception, match="test_loader"):
        resolve_evaluate_config(config="invalid")


def test_resolve_apply_config_fills_local_config_saving_fallbacks(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: auto\n")
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="folder_outside",
        INFERENCE_SAVE_INPUT_MODE="if_not_in_place",
    )

    resolved = resolve_apply_config(stg=local)

    assert resolved.saving.mode == "folder_outside"
    assert resolved.saving.save_input_mode == "if_not_in_place"


def test_resolve_apply_save_input_interprets_resolved_policy():
    in_place_cfg = ApplyDefaults.model_validate(
        {"saving": {"mode": "in_place", "save_input_mode": "if_not_in_place"}}
    )
    default_cfg = ApplyDefaults.model_validate(
        {"saving": {"mode": "default", "save_input_mode": "if_not_in_place"}}
    )

    in_place = defaults_mod.resolve_apply_output_policy(in_place_cfg)
    default = defaults_mod.resolve_apply_output_policy(default_cfg)

    assert resolve_apply_save_input(in_place_cfg, output_policy=in_place) is False
    assert resolve_apply_save_input(default_cfg, output_policy=default) is True


def test_named_saving_override_overrides_local_config_policy(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: auto\n")
    _write(
        inference_config_dir / "local" / "no_input.yml",
        "apply:\n  saving:\n    save_input_mode: never\n",
    )
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="default",
        INFERENCE_SAVE_INPUT_MODE="always",
    )

    resolved = resolve_apply_config(config="no_input", stg=local)
    policy = defaults_mod.resolve_apply_output_policy(resolved)

    assert resolved.saving.save_input_mode == "never"
    assert resolve_apply_save_input(resolved, output_policy=policy) is False


def test_typed_save_input_override_has_priority(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: auto\n")
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="in_place",
        INFERENCE_SAVE_INPUT_MODE="never",
    )

    resolved = resolve_apply_config(
        overrides=ApplyOverrides.model_validate(
            {"saving": {"save_input_mode": "always"}}
        ),
        stg=local,
    )
    policy = defaults_mod.resolve_apply_output_policy(resolved)

    assert resolved.saving.save_input_mode == "always"
    assert resolve_apply_save_input(resolved, output_policy=policy) is True


def test_apply_saving_config_overrides_local_config_route(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_inside\n")
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="folder_outside",
        INFERENCE_SAVE_INPUT_MODE="if_not_in_place",
    )

    resolved = resolve_apply_config(stg=local)
    policy = defaults_mod.resolve_apply_output_policy(resolved)

    assert resolved.saving.mode == "folder_inside"
    assert policy.mode == "folder_inside"


def test_named_local_saving_override_has_priority_over_local_defaults(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_inside\n")
    _write(inference_config_dir / "local" / "special.yml", "apply:\n  saving:\n    mode: in_place\n")
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="folder_outside",
        INFERENCE_SAVE_INPUT_MODE="if_not_in_place",
    )

    resolved = resolve_apply_config(config="special", stg=local)
    policy = defaults_mod.resolve_apply_output_policy(resolved)

    assert policy.mode == "in_place"


def test_named_nonlocal_saving_override_has_priority_over_local_defaults(
    inference_config_dir: Path,
):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_inside\n")
    _write(
        inference_config_dir / "examples" / "portable.yml",
        "apply:\n  saving:\n    mode: folder_outside\n",
    )
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="in_place",
        INFERENCE_SAVE_INPUT_MODE="if_not_in_place",
    )

    resolved = resolve_apply_config(config="examples/portable", stg=local)
    policy = defaults_mod.resolve_apply_output_policy(resolved)

    assert policy.mode == "folder_outside"


def test_typed_output_override_has_highest_priority(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "apply:\n  saving:\n    mode: folder_outside\n")
    local = SimpleNamespace(
        INFERENCE_OUTPUT_MODE="in_place",
        INFERENCE_SAVE_INPUT_MODE="if_not_in_place",
    )

    resolved = resolve_apply_config(
        overrides=ApplyOverrides.model_validate(
            {"saving": {"mode": "folder_inside"}}
        ),
        stg=local,
    )
    policy = defaults_mod.resolve_apply_output_policy(resolved)

    assert policy.mode == "folder_inside"


def test_builtin_post_training_preset_is_used_if_local_file_is_missing(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  inference:\n    tiling_size: 256\n")

    resolved = resolve_evaluate_config(config="post_training")

    assert resolved.inference.tiling_size == 256
    assert resolved.checkpoint.best_or_last == "both"
    assert resolved.metrics == ["psnr", "ssim"]
    assert resolved.saving.overwrite is True


def test_local_post_training_overrides_builtin_and_inherits_local_defaults(inference_config_dir: Path):
    _write_local_defaults(inference_config_dir, "evaluate:\n  inference:\n    tiling_size: 384\n")
    _write(
        inference_config_dir / "local" / "post_training.yml",
        "evaluate:\n  metrics:\n    - custom_metric\n",
    )

    resolved = resolve_evaluate_config(config="post_training")

    assert resolved.inference.tiling_size == 384
    assert resolved.checkpoint.best_or_last == "both"
    assert resolved.metrics == ["custom_metric"]
    assert resolved.saving.overwrite is True


def test_promoted_model_config_layers_between_local_defaults_and_explicit_config(
    inference_config_dir: Path,
):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    tiling_size: 256\n    lvae_num_samples: 30\n",
    )
    model_config = inference_config_dir / "model_defaults.yml"
    _write(model_config, "apply:\n  inference:\n    lvae_num_samples: 10\n")
    _write(
        inference_config_dir / "local" / "large_tiles.yml",
        "apply:\n  inference:\n    tiling_size: 512\n",
    )

    resolved = resolve_apply_config(
        model_config=model_config,
        config="large_tiles",
    )

    assert resolved.inference.lvae_num_samples == 10
    assert resolved.inference.tiling_size == 512


def test_explicit_config_and_cli_override_promoted_model_config_with_warning(
    inference_config_dir: Path,
):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    lvae_num_samples: 30\n",
    )
    model_config = inference_config_dir / "model_defaults.yml"
    _write(model_config, "apply:\n  inference:\n    lvae_num_samples: 10\n")
    selected_config = inference_config_dir / "selected.yml"
    _write(selected_config, "apply:\n  inference:\n    lvae_num_samples: 20\n")

    with pytest.warns(UserWarning, match=r"lvae_num_samples=10.*CLI override.*5.*Using 5"):
        resolved = resolve_apply_config(
            model_config=model_config,
            config=selected_config,
            overrides=ApplyOverrides(
                inference=ApplyInferenceOverrides(lvae_num_samples=5),
            ),
        )

    assert resolved.inference.lvae_num_samples == 5


def test_explicit_config_override_promoted_model_config_warns_without_cli_override(
    inference_config_dir: Path,
):
    _write_local_defaults(inference_config_dir, "apply:\n  inference:\n    tiling_size: 256\n")
    model_config = inference_config_dir / "model_defaults.yml"
    _write(model_config, "apply:\n  inference:\n    lvae_num_samples: 10\n")
    selected_config = inference_config_dir / "selected.yml"
    _write(selected_config, "apply:\n  inference:\n    lvae_num_samples: 20\n")

    with pytest.warns(UserWarning, match=r"lvae_num_samples=10.*selected.yml.*20.*Using 20"):
        resolved = resolve_apply_config(
            model_config=model_config,
            config=selected_config,
        )

    assert resolved.inference.lvae_num_samples == 20


def test_local_defaults_do_not_warn_when_promoted_model_overrides_them(
    inference_config_dir: Path,
):
    _write_local_defaults(
        inference_config_dir,
        "apply:\n  inference:\n    lvae_num_samples: 30\n",
    )
    model_config = inference_config_dir / "model_defaults.yml"
    _write(model_config, "apply:\n  inference:\n    lvae_num_samples: 10\n")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        resolved = resolve_apply_config(model_config=model_config)

    assert captured == []
    assert resolved.inference.lvae_num_samples == 10
