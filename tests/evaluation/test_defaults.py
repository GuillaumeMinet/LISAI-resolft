from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.evaluation.defaults as defaults_mod
from lisai.config.io.config_paths import ConfigPathResolver
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
        INFERENCE_DEFAULT_CONFIG_NAME="defaults",
        CONFIG_SUFFIXES=(".yml", ".yaml"),
    )
    monkeypatch.setattr(
        defaults_mod,
        "inference_config_paths",
        ConfigPathResolver("inference", stg=fake_settings),
    )
    return config_dir


def test_resolve_inference_config_path_defaults_to_configs_inference_defaults(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    _write(defaults_path, "apply:\n  tiling_size: 256\n")

    resolved = resolve_inference_config_path(None)

    assert resolved == defaults_path.resolve()


def test_resolve_apply_options_merges_defaults_then_named_config_then_cli(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    fast_path = inference_config_dir / "fast_upsamp.yml"
    _write(
        defaults_path,
        """
apply:
  tiling_size: 256
  denormalize_output: false
  fill_factor: 0.5
  color_code_prm:
    colormap: turbo
    saturation: 0.35
evaluate:
  split: test
""".strip()
        + "\n",
    )
    _write(
        fast_path,
        """
apply:
  tiling_size: 512
  crop_size: 128
  fill_factor: 0.75
""".strip()
        + "\n",
    )

    resolved = resolve_apply_options(config="fast_upsamp")

    assert resolved["tiling_size"] == 512
    assert resolved["crop_size"] == 128
    assert resolved["fill_factor"] == pytest.approx(0.75)
    assert resolved["denormalize_output"] is False
    assert "save_inp" not in resolved
    assert resolved["color_code_prm"]["colormap"] == "turbo"


def test_resolve_apply_options_preserves_legacy_downsamp_when_fill_factor_is_not_set(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    _write(
        defaults_path,
        """
apply:
  downsamp: 2
  fill_factor: null
""".strip()
        + "\n",
    )

    resolved = resolve_apply_options()

    assert resolved["downsamp"] == 2
    assert resolved["fill_factor"] is None


def test_resolve_evaluate_options_preserves_tiling_policy_values(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    no_tiling_path = inference_config_dir / "no_tiling.yml"
    _write(defaults_path, "evaluate:\n  tiling_size: auto\n")
    _write(no_tiling_path, "evaluate:\n  tiling_size: off\n")

    resolved = resolve_evaluate_options(config="no_tiling")
    forced = resolve_evaluate_options(config="no_tiling", tiling_size=512)

    assert resolved["tiling_size"] == "off"
    assert forced["tiling_size"] == 512


def test_resolve_evaluate_options_requires_requested_section_in_named_config(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    apply_only_path = inference_config_dir / "apply_only.yml"
    _write(defaults_path, "evaluate:\n  split: test\n")
    _write(apply_only_path, "apply:\n  tiling_size: 512\n")

    with pytest.raises(ValueError, match="does not define a 'evaluate' section"):
        resolve_evaluate_options(config="apply_only")


def test_inference_config_rejects_unknown_evaluate_keys(inference_config_dir: Path):
    defaults_path = inference_config_dir / "defaults.yml"
    invalid_path = inference_config_dir / "invalid.yml"
    _write(defaults_path, "evaluate:\n  split: test\n")
    _write(invalid_path, "evaluate:\n  test_loader: bad\n")

    with pytest.raises(Exception, match="test_loader"):
        resolve_evaluate_options(config="invalid")


def test_resolve_apply_save_input_uses_local_policy(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    in_place = defaults_mod.ApplyOutputPolicy(mode="in_place")
    default = defaults_mod.ApplyOutputPolicy(mode="default")
    local = SimpleNamespace(INFERENCE_SAVE_INPUT_MODE="if_not_in_place")

    assert resolve_apply_save_input(output_policy=in_place, stg=local) is False
    assert resolve_apply_save_input(output_policy=default, stg=local) is True


def test_named_save_input_mode_overrides_local_policy(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    _write(
        inference_config_dir / "no_input.yml",
        "apply:\n  output:\n    save_input_mode: never\n",
    )
    policy = defaults_mod.ApplyOutputPolicy(mode="default")
    local = SimpleNamespace(INFERENCE_SAVE_INPUT_MODE="always")

    assert resolve_apply_save_input(
        output_policy=policy, config="no_input", stg=local
    ) is False


def test_cli_save_input_override_has_priority(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    policy = defaults_mod.ApplyOutputPolicy(mode="in_place")
    local = SimpleNamespace(INFERENCE_SAVE_INPUT_MODE="never")

    assert resolve_apply_save_input(
        output_policy=policy, save_input=True, stg=local
    ) is True
    assert resolve_apply_save_input(
        output_policy=policy, save_input=False, stg=local
    ) is False


def test_resolve_apply_output_policy_uses_local_default_when_no_override(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="default")

    policy = defaults_mod.resolve_apply_output_policy(stg=local)

    assert policy.mode == "default"
    assert policy.save_folder is None


def test_resolve_apply_output_policy_uses_local_in_place_when_no_override(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(stg=local)

    assert policy.mode == "in_place"


def test_resolve_apply_output_policy_uses_source_relative_local_modes(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")

    folder_inside = defaults_mod.resolve_apply_output_policy(
        stg=SimpleNamespace(INFERENCE_OUTPUT_MODE="folder_inside")
    )
    folder_outside = defaults_mod.resolve_apply_output_policy(
        stg=SimpleNamespace(INFERENCE_OUTPUT_MODE="folder_outside")
    )

    assert folder_inside.mode == "folder_inside"
    assert folder_outside.mode == "folder_outside"


def test_named_inference_output_overrides_local_preference(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    _write(
        inference_config_dir / "upsamp.yml",
        "apply:\n  output:\n    in_place: false\n  tiling_size: 512\n",
    )
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(config="upsamp", stg=local)

    assert policy.mode == "default"


def test_named_inference_output_mode_overrides_local_preference(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    _write(
        inference_config_dir / "folder_inside.yml",
        "apply:\n  output:\n    mode: folder_inside\n",
    )
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="folder_outside")

    policy = defaults_mod.resolve_apply_output_policy(config="folder_inside", stg=local)

    assert policy.mode == "folder_inside"


def test_named_inference_save_folder_overrides_local_preference(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    _write(
        inference_config_dir / "special.yml",
        "apply:\n  output:\n    save_folder: /tmp/special_predictions\n",
    )
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(config="special", stg=local)

    assert policy.mode == "folder"
    assert policy.save_folder == Path("/tmp/special_predictions")


def test_cli_output_mode_has_priority_over_named_config(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    _write(
        inference_config_dir / "special.yml",
        "apply:\n  output:\n    mode: folder_outside\n",
    )
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(
        config="special",
        output_mode="folder_inside",
        stg=local,
    )

    assert policy.mode == "folder_inside"


def test_cli_output_override_has_priority_over_named_config(inference_config_dir: Path):
    _write(inference_config_dir / "defaults.yml", "apply:\n  tiling_size: auto\n")
    _write(
        inference_config_dir / "special.yml",
        "apply:\n  output:\n    in_place: true\n",
    )
    local = SimpleNamespace(INFERENCE_OUTPUT_MODE="in_place")

    policy = defaults_mod.resolve_apply_output_policy(
        config="special",
        in_place=False,
        stg=local,
    )

    assert policy.mode == "default"
