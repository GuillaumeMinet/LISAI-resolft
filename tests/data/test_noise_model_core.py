"""Patch A regression: extraction must retain the legacy HDN calibration numerics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from lisai.data.noise_model import (
    NoiseModelBuildConfig,
    NoiseModelCheckConfig,
    build_noise_model,
    check_noise_model,
)
from lisai.data.noise_model.core import (
    _prepare_signal_observation_arrays,
    clip_crop_pair,
    make_noise_histogram,
    prepare_noise_model_data,
)
from lisai.models.load_nm import load_noise_model_from_paths


class _Paths:
    """Small path adapter: the legacy builder receives Paths explicitly."""

    def __init__(self, root: Path):
        self.root = root

    def dataset_dir(self, *, dataset_name: str) -> Path:
        return self.root / "datasets" / dataset_name

    def noise_model_dir(self, *, noiseModel_name: str) -> Path:
        return self.root / "noise_models" / noiseModel_name

    def noise_model_path(self, *, noiseModel_name: str) -> Path:
        return self.noise_model_dir(noiseModel_name=noiseModel_name) / "GMM.npz"


def _make_data(paths: _Paths, name: str) -> None:
    source = paths.dataset_dir(dataset_name=name)
    (source / "signal").mkdir(parents=True)
    (source / "observation").mkdir(parents=True)
    y, x = np.mgrid[:10, :12]
    for i in range(3):
        signal = (2 + 0.65 * x + 0.3 * y + i + np.cos((x + y + i) * 1.1)).astype("float32")
        observations = np.stack(
            [
                signal * 0.6 + 2 + np.sin(y * 2),
                signal * 0.8 + 3 + np.cos(x * 2),
                signal * 1.25 + 1 + np.sin(x + y),
            ],
        ).astype("float32")
        tifffile.imwrite(source / "signal" / f"{i:02}.tif", signal)
        tifffile.imwrite(
            source / "observation" / f"{i:02}.tif", observations, photometric="minisblack"
        )


@pytest.mark.parametrize(
    ("name", "noise_level", "norm_sig_to_obs", "expected_min", "expected_max"),
    [
        (
            "single", 1, True, -2.4895682334899902, 2.211885690689087,
        ),
        (
            "multi", [0, 2], True, -1.9177870750427246, 2.7119359970092773,
        ),
        (
            "multi_no_norm", [0, 2], False, -2.4895682334899902, 2.211885929107666,
        ),
    ],
)
def test_builder_matches_original_fit_and_artifacts(
    tmp_path, name, noise_level, norm_sig_to_obs, expected_min, expected_max
):
    """Stable preparation golden values from the original a76d2ad implementation.

    Exact optimizer weights vary slightly with Torch runtime/CPU math libraries.
    File-vs-array seeded numerical equivalence is tested separately in
    test_noise_model_api.py; here we validate the persisted GMM contract.
    """
    paths = _Paths(tmp_path)
    _make_data(paths, name)
    cfg = NoiseModelBuildConfig(
        dataset_name=name,
        signal_subfolder="signal",
        observation_subfolder="observation",
        noise_level=noise_level,
        norm_sig_to_obs=norm_sig_to_obs,
        normalize_everything=True,
        clip=-3.0,
        crop_size=(8, 10),
        histogram_bins=16,
        gmm_n_gaussian=1,
        gmm_n_coeff=2,
        gmm_n_epochs=4,
        gmm_batch_size=100,
        gmm_min_sigma=0.1,
        save_name=name,
        save_gmm_plot=False,
    )
    np.random.seed(71)
    torch.manual_seed(71)
    result = build_noise_model(cfg, paths=paths, device=torch.device("cpu"))
    assert result.save_dir == paths.noise_model_dir(noiseModel_name=name)
    n = 3 if isinstance(noise_level, int) else 6
    assert result.signal_shape == (n, 8, 10)
    assert result.observation_shape == (n, 8, 10)
    assert result.min_signal == pytest.approx(expected_min, abs=2e-5)
    assert result.max_signal == pytest.approx(expected_max, abs=2e-5)

    hist = np.load(result.histogram_path)
    assert hist.shape == (3, 16, 16)
    np.testing.assert_allclose(hist[0].sum(axis=1), 1.0, atol=1e-12)
    with np.load(result.gmm_path) as saved:
        assert set(saved.files) == {"trained_weight", "min_signal", "max_signal", "min_sigma"}
        assert saved["trained_weight"].shape == (3, 2)
        assert np.isfinite(saved["trained_weight"]).all()
    norms = json.loads(result.norm_prm_path.read_text(encoding="utf-8"))
    assert norms["clip"] == -3.0
    assert norms["normalize_data"] is True
    assert norms["normSig2Obs"] is norm_sig_to_obs
    if isinstance(noise_level, list):
        assert len(norms["avgObs_per_noise"]) == len(noise_level)
        assert len(norms["stdObs_per_noise"]) == len(noise_level)
    else:
        assert "avgObs_per_noise" not in norms
    model, loaded_norms = load_noise_model_from_paths(
        result.gmm_path, result.norm_prm_path, torch.device("cpu")
    )
    assert model.n_gaussian == 1
    assert loaded_norms == norms
    if name == "single":
        checked = check_noise_model(
            NoiseModelCheckConfig(
                dataset_name=name,
                signal_subfolder="signal",
                observation_subfolder="observation",
                noise_model_name=name,
                noise_level=1,
                norm_sig_to_obs=True,
                normalize_everything=True,
                clip=-3.0,
                crop_size=(8, 10),
                histogram_bins=16,
                display=False,
                save_plot=False,
            ),
            paths=paths,
            device=torch.device("cpu"),
        )
        assert checked.noise_model_path == result.gmm_path
        assert checked.signal_shape == (3, 8, 10)


def test_array_preparation_preserves_original_multi_level_order_and_stats():
    signal = [np.arange(12, dtype=np.float32).reshape(3, 4) + i for i in (0, 2)]
    observation = [
        np.stack([sig + 10, sig * 2 + 20], axis=0) for sig in signal
    ]
    originals = [(sig.copy(), obs.copy()) for sig, obs in zip(signal, observation)]
    prepared = prepare_noise_model_data(
        signal,
        observation,
        noise_level=[0, 1],
        norm_sig_to_obs=True,
        normalize_everything=False,
        clip=None,
    )
    # The legacy Fortran reshape groups the level-0 images before level-1 images.
    np.testing.assert_array_equal(
        prepared.observation, np.stack([observation[0][0], observation[1][0],
                                        observation[0][1], observation[1][1]])
    )
    assert prepared.norm_prm["avgObs_per_noise"] == pytest.approx(
        [float(np.mean(np.stack([o[0] for o in observation]))),
         float(np.mean(np.stack([o[1] for o in observation])))]
    )
    assert prepared.norm_prm["normalize_data"] is False
    histogram = make_noise_histogram(prepared, bins=12)
    assert histogram.shape == (3, 12, 12)
    for (old_sig, old_obs), sig, obs in zip(originals, signal, observation):
        np.testing.assert_array_equal(sig, old_sig)
        np.testing.assert_array_equal(obs, old_obs)


def test_multi_noise_without_signal_matching_keeps_legacy_repetition_order():
    signal = [np.ones((2, 3)) * 2, np.ones((2, 3)) * 3]
    observation = [np.stack([s + 1, s + 2]) for s in signal]
    s, o, stats = _prepare_signal_observation_arrays(
        signal, observation, noise_level=[0, 1], norm_sig_to_obs=False
    )
    np.testing.assert_array_equal(s, np.stack([signal[0], signal[0], signal[1], signal[1]]))
    np.testing.assert_array_equal(o, np.concatenate(observation))
    assert stats == {}


def test_clip_crop_pair_does_not_mutate_sources():
    signal = np.arange(24, dtype=np.float32).reshape(4, 6) - 10
    observation = np.stack([signal * 2, signal * 3])
    original_signal, original_observation = signal.copy(), observation.copy()
    new_signal, new_obs = clip_crop_pair(signal, observation, clip=-3, crop_size=(2, 4))
    np.testing.assert_array_equal(new_signal, np.maximum(signal, -3)[1:3, 1:5])
    np.testing.assert_array_equal(new_obs, np.maximum(observation, -3)[:, 1:3, 1:5])
    np.testing.assert_array_equal(signal, original_signal)
    np.testing.assert_array_equal(observation, original_observation)
