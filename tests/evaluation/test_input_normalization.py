"""Apply-only input normalization alternatives for existing checkpoints."""

import numpy as np
import pytest
from pydantic import ValidationError

from lisai.config.models.inference import ApplyDefaults, ApplyOverrides, InputNormalization
from lisai.evaluation.inference.normalization import (
    normalize_apply_input,
    normalize_inp,
)


def test_training_mode_is_identical_to_existing_normalizer():
    image = np.array([[-10, -3, 3, 30]], dtype=np.float32)
    norm = {"normalize_data": True, "avgObs": 10.0, "stdObs": 5.0}
    model = {"data_mean": 0.5, "data_std": 2.0}
    expected = normalize_inp(image.copy(), -3, norm, model)

    actual = normalize_apply_input(image.copy(), -3, norm, model, InputNormalization())
    np.testing.assert_array_equal(actual, expected)


def test_manual_mode_adjusts_intensity_before_legacy_clipping():
    image = np.array([[-2, 0, 2]], dtype=np.float32)
    model = {"data_mean": 5.0, "data_std": 2.0}
    cfg = InputNormalization(mode="manual", gain=10.0, offset=-7.0)

    actual = normalize_apply_input(image, -3, None, model, cfg)
    # preclip: [-27, -7, 13]; postclip: [-3, -3, 13]
    np.testing.assert_array_equal(actual, np.array([[-4, -4, 4]], dtype=np.float32))
    np.testing.assert_array_equal(image, [[-2, 0, 2]])  # source not mutated


def test_image_mode_matches_single_stage_model_reference():
    image = np.array([[1, 3], [5, 7]], dtype=np.float32)
    model = {"data_mean": 25.0, "data_std": 40.0}

    actual = normalize_apply_input(
        image, False, None, model, InputNormalization(mode="image")
    )
    np.testing.assert_allclose(actual.mean(), 0, atol=1e-6)
    np.testing.assert_allclose(actual.std(), 1, atol=1e-6)
    np.testing.assert_array_equal(image, [[1, 3], [5, 7]])


def test_image_mode_handles_two_stage_hdn_observation_and_model_stats():
    image = np.array([[-2, 0], [3, 7]], dtype=np.float32)
    noise_norm = {"normalize_data": True, "avgObs": 18.0, "stdObs": 35.0}
    model_norm = {"data_mean": 0.25, "data_std": 2.0}

    actual = normalize_apply_input(
        image, False, noise_norm, model_norm, InputNormalization(mode="image")
    )
    np.testing.assert_allclose(actual.mean(), 0, atol=1e-6)
    np.testing.assert_allclose(actual.std(), 1, atol=1e-6)


def test_image_mode_uses_shared_statistics_for_all_frames():
    stack = np.stack(
        [np.full((2, 2), 2.0, dtype=np.float32), np.full((2, 2), 8.0, dtype=np.float32)]
    )
    actual = normalize_apply_input(
        stack, False, None, {"data_mean": 15., "data_std": 10.},
        InputNormalization(mode="image"),
    )
    np.testing.assert_allclose(actual[0], -1, atol=1e-6)
    np.testing.assert_allclose(actual[1], 1, atol=1e-6)


@pytest.mark.parametrize("image, data_norm, model_norm, message", [
    (np.ones((2, 2)), None, {"data_mean": 0., "data_std": 1.}, "variance"),
    (np.array([0., np.nan]), None, {"data_mean": 0., "data_std": 1.}, "finite"),
    (np.array([0., 1.]), None, None, "saved training"),
    (np.array([0., 1.]), None, {"data_mean": 0., "data_std": 0.}, "invalid"),
])
def test_image_mode_rejects_invalid_inputs(image, data_norm, model_norm, message):
    with pytest.raises(ValueError, match=message):
        normalize_apply_input(
            image, False, data_norm, model_norm, InputNormalization(mode="image")
        )


def test_typed_defaults_and_sparse_yaml_overrides():
    assert ApplyDefaults().input.normalization.mode == "training"
    overrides = ApplyOverrides.model_validate(
        {"input": {"normalization": {"mode": "manual", "gain": 9.94}}}
    )
    assert overrides.model_dump(exclude_unset=True) == {
        "input": {"normalization": {"mode": "manual", "gain": 9.94}}
    }


@pytest.mark.parametrize("payload", [
    {"mode": "auto"},
    {"mode": "manual", "gain": 0},
    {"mode": "manual", "gain": float("inf")},
    {"mode": "manual", "offset": float("nan")},
    {"mode": "manual", "bogus": 9},
])
def test_config_rejects_invalid_modes_and_parameters(payload):
    with pytest.raises(ValidationError):
        InputNormalization.model_validate(payload)
