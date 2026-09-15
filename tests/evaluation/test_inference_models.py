from __future__ import annotations

import pytest
from pydantic import ValidationError

from lisai.config.models.inference import (
    InferenceConfig,
    InferenceDefaults,
    InferenceOverrides,
    ResolvedInferenceConfig,
    TilingSizePolicy,
)
from lisai.config.models.inference_defaults import (
    InferenceConfig as LegacyInferenceConfig,
)
from lisai.config.models.inference_defaults import (
    InferenceDefaults as LegacyInferenceDefaults,
)
from lisai.config.models.inference_defaults import (
    InferenceOverrides as LegacyInferenceOverrides,
)
from lisai.config.models.inference_defaults import (
    ResolvedInferenceConfig as LegacyResolvedInferenceConfig,
)
from lisai.config.models.inference_defaults import (
    TilingSizePolicy as LegacyTilingSizePolicy,
)


def test_legacy_inference_defaults_module_reexports_new_root_models():
    assert LegacyInferenceConfig is InferenceConfig
    assert LegacyInferenceDefaults is InferenceDefaults
    assert LegacyInferenceOverrides is InferenceOverrides
    assert LegacyResolvedInferenceConfig is ResolvedInferenceConfig
    assert LegacyTilingSizePolicy is TilingSizePolicy


def test_backward_compatibility_aliases_point_to_clearer_names():
    assert InferenceConfig is InferenceOverrides
    assert InferenceDefaults is ResolvedInferenceConfig


def test_inference_models_accept_apply_fill_factor_when_null_or_valid():
    null_fill = InferenceOverrides.model_validate({"apply": {"inference": {"fill_factor": None}}})
    valid_fill = InferenceOverrides.model_validate({"apply": {"inference": {"fill_factor": 0.5}}})

    assert null_fill.apply is not None
    assert null_fill.apply.inference is not None
    assert null_fill.apply.inference.fill_factor is None
    assert valid_fill.apply is not None
    assert valid_fill.apply.inference is not None
    assert valid_fill.apply.inference.fill_factor == pytest.approx(0.5)


def test_flat_legacy_apply_layout_is_migrated_to_nested_sections():
    cfg = InferenceOverrides.model_validate(
        {
            "apply": {
                "tiling_size": 512,
                "denormalize_output": False,
                "apply_color_code": True,
                "color_code_prm": {"colormap": "viridis"},
                "output": {"mode": "folder_inside"},
            }
        }
    )

    assert cfg.apply is not None
    assert cfg.apply.inference is not None
    assert cfg.apply.inference.tiling_size == 512
    assert cfg.apply.postprocess is not None
    assert cfg.apply.postprocess.denormalize is False
    assert cfg.apply.postprocess.color_code is not None
    assert cfg.apply.postprocess.color_code.enabled is True
    assert cfg.apply.postprocess.color_code.colormap == "viridis"
    assert cfg.apply.saving is not None
    assert cfg.apply.saving.mode == "folder_inside"


def test_flat_legacy_evaluate_layout_is_migrated_to_nested_sections():
    cfg = InferenceOverrides.model_validate(
        {
            "evaluate": {
                "tiling_size": "off",
                "metrics_list": ["psnr"],
                "data_prm_update": {"subfolder": "test"},
                "overwrite": True,
            }
        }
    )

    assert cfg.evaluate is not None
    assert cfg.evaluate.inference is not None
    assert cfg.evaluate.inference.tiling_size == "off"
    assert cfg.evaluate.metrics == ["psnr"]
    assert cfg.evaluate.data is not None
    assert cfg.evaluate.data.overrides == {"subfolder": "test"}
    assert cfg.evaluate.saving is not None
    assert cfg.evaluate.saving.overwrite is True


def test_inference_models_accept_tiling_policy_values():
    defaults = ResolvedInferenceConfig.model_validate(
        {"evaluate": {"inference": {"tiling_size": "auto"}}}
    )
    no_tiling = InferenceOverrides.model_validate(
        {"evaluate": {"inference": {"tiling_size": "off"}}}
    )
    forced = InferenceOverrides.model_validate(
        {"apply": {"inference": {"tiling_size": 2000}}}
    )
    legacy_null = InferenceOverrides.model_validate({"apply": {"tiling_size": None}})

    assert defaults.evaluate.inference.tiling_size == "auto"
    assert no_tiling.evaluate is not None
    assert no_tiling.evaluate.inference is not None
    assert no_tiling.evaluate.inference.tiling_size == "off"
    assert forced.apply is not None
    assert forced.apply.inference is not None
    assert forced.apply.inference.tiling_size == 2000
    assert legacy_null.apply is not None
    assert legacy_null.apply.inference is not None
    assert legacy_null.apply.inference.tiling_size is None


def test_inference_models_reject_invalid_apply_fill_factor():
    with pytest.raises(ValidationError, match="fill_factor"):
        InferenceOverrides.model_validate({"apply": {"inference": {"fill_factor": 0}}})

    with pytest.raises(ValidationError, match="fill_factor"):
        InferenceOverrides.model_validate({"apply": {"inference": {"fill_factor": 1.2}}})


def test_inference_models_reject_invalid_tiling_policy_values():
    with pytest.raises(ValidationError, match="tiling_size"):
        InferenceOverrides.model_validate({"evaluate": {"inference": {"tiling_size": "large"}}})

    with pytest.raises(ValidationError, match="tiling_size"):
        InferenceOverrides.model_validate({"evaluate": {"inference": {"tiling_size": 0}}})


def test_apply_saving_override_is_optional_and_sparse():
    no_saving = InferenceOverrides.model_validate({"apply": {"inference": {"tiling_size": 512}}})
    folder_inside = InferenceOverrides.model_validate({"apply": {"saving": {"mode": "folder_inside"}}})
    folder_outside = InferenceOverrides.model_validate({"apply": {"saving": {"mode": "folder_outside"}}})
    in_place = InferenceOverrides.model_validate({"apply": {"saving": {"in_place": True}}})
    force_default = InferenceOverrides.model_validate({"apply": {"saving": {"in_place": False}}})

    assert no_saving.apply is not None
    assert no_saving.apply.saving is None
    assert folder_inside.apply is not None
    assert folder_inside.apply.saving is not None
    assert folder_inside.apply.saving.mode == "folder_inside"
    assert folder_outside.apply is not None
    assert folder_outside.apply.saving is not None
    assert folder_outside.apply.saving.mode == "folder_outside"
    assert in_place.apply is not None
    assert in_place.apply.saving is not None
    assert in_place.apply.saving.in_place is True
    assert force_default.apply is not None
    assert force_default.apply.saving is not None
    assert force_default.apply.saving.in_place is False


def test_apply_saving_override_accepts_save_input_mode_with_destination():
    cfg = InferenceOverrides.model_validate(
        {
            "apply": {
                "saving": {
                    "mode": "in_place",
                    "save_input_mode": "always",
                }
            }
        }
    )

    assert cfg.apply is not None
    assert cfg.apply.saving is not None
    assert cfg.apply.saving.mode == "in_place"
    assert cfg.apply.saving.save_input_mode == "always"


def test_apply_saving_override_rejects_retired_mode_names():
    for mode in ("inside", "next_to"):
        with pytest.raises(ValidationError, match="mode"):
            InferenceOverrides.model_validate({"apply": {"saving": {"mode": mode}}})


def test_apply_saving_override_rejects_conflicting_destination_fields():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        InferenceOverrides.model_validate(
            {
                "apply": {
                    "saving": {
                        "mode": "folder_inside",
                        "save_folder": "/tmp/predictions",
                    }
                }
            }
        )
