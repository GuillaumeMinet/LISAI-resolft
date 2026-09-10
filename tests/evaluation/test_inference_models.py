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
    null_fill = InferenceOverrides.model_validate({"apply": {"fill_factor": None}})
    valid_fill = InferenceOverrides.model_validate({"apply": {"fill_factor": 0.5}})

    assert null_fill.apply is not None
    assert null_fill.apply.fill_factor is None
    assert valid_fill.apply is not None
    assert valid_fill.apply.fill_factor == pytest.approx(0.5)


def test_inference_models_accept_tiling_policy_values():
    defaults = ResolvedInferenceConfig.model_validate({"evaluate": {"tiling_size": "auto"}})
    no_tiling = InferenceOverrides.model_validate({"evaluate": {"tiling_size": "off"}})
    forced = InferenceOverrides.model_validate({"apply": {"tiling_size": 2000}})
    legacy_null = InferenceOverrides.model_validate({"apply": {"tiling_size": None}})

    assert defaults.evaluate.tiling_size == "auto"
    assert no_tiling.evaluate is not None
    assert no_tiling.evaluate.tiling_size == "off"
    assert forced.apply is not None
    assert forced.apply.tiling_size == 2000
    assert legacy_null.apply is not None
    assert legacy_null.apply.tiling_size is None


def test_inference_models_reject_invalid_apply_fill_factor():
    with pytest.raises(ValidationError, match="fill_factor"):
        InferenceOverrides.model_validate({"apply": {"fill_factor": 0}})

    with pytest.raises(ValidationError, match="fill_factor"):
        InferenceOverrides.model_validate({"apply": {"fill_factor": 1.2}})


def test_inference_models_reject_invalid_tiling_policy_values():
    with pytest.raises(ValidationError, match="tiling_size"):
        InferenceOverrides.model_validate({"evaluate": {"tiling_size": "large"}})

    with pytest.raises(ValidationError, match="tiling_size"):
        InferenceOverrides.model_validate({"evaluate": {"tiling_size": 0}})
