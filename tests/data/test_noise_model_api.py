"""Public NumPy -> workspace noise-model creation, on top of Patch A's numerical core."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import torch

from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.data.noise_model import NoiseModelBuildConfig, build_noise_model
from lisai.infra.paths import Paths
from lisai.models.load_nm import load_noise_model_from_paths
from lisai.noise_models import NoiseModelOptions, NoiseModelResult, create_noise_model


@pytest.fixture
def settings(tmp_path):
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    return Settings.from_workspace(workspace)


@pytest.fixture
def sample_pairs():
    y, x = np.mgrid[:12, :14]
    signal = np.stack(
        [3 + 0.6 * x + 0.4 * y + i + np.sin((x + y + i) * 1.1) for i in range(3)]
    ).astype(np.float32)
    observation = (signal * 0.7 + 3 + np.cos(y * 2)).astype(np.float32)
    return signal, observation


@pytest.fixture
def options():
    return NoiseModelOptions(
        histogram_bins=16,
        gmm_n_gaussian=1,
        gmm_n_coeff=2,
        gmm_n_epochs=4,
        gmm_batch_size=100,
        gmm_min_sigma=0.1,
        crop_size=(10, 12),
        save_gmm_plot=False,
    )


def _create(settings, sample_pairs, options, *, name="calibrated", **kwargs):
    return create_noise_model(
        *sample_pairs, name=name, settings=settings, options=options, device="cpu", **kwargs
    )


def test_array_api_matches_legacy_builder_and_loads_for_hdn(settings, sample_pairs, options):
    signal, observation = sample_pairs
    input_signal, input_observation = signal.copy(), observation.copy()
    paths = Paths(settings)
    ds_dir = paths.dataset_dir(dataset_name="calibration")
    (ds_dir / "signal").mkdir(parents=True)
    (ds_dir / "observation").mkdir()
    for i, (sig, obs) in enumerate(zip(signal, observation)):
        tifffile.imwrite(ds_dir / "signal" / f"{i:02}.tif", sig)
        tifffile.imwrite(ds_dir / "observation" / f"{i:02}.tif", obs)

    np.random.seed(73)
    torch.manual_seed(73)
    legacy = build_noise_model(
        NoiseModelBuildConfig(
            dataset_name="calibration", signal_subfolder="signal",
            observation_subfolder="observation", save_name="legacy",
            histogram_bins=options.histogram_bins, crop_size=options.crop_size,
            gmm_n_gaussian=options.gmm_n_gaussian, gmm_n_coeff=options.gmm_n_coeff,
            gmm_n_epochs=options.gmm_n_epochs, gmm_batch_size=options.gmm_batch_size,
            gmm_min_sigma=options.gmm_min_sigma, save_gmm_plot=False,
        ), paths=paths, device=torch.device("cpu"),
    )
    np.random.seed(73)
    torch.manual_seed(73)
    public = _create(settings, sample_pairs, options)
    assert isinstance(public, NoiseModelResult)
    assert public.name == "calibrated"
    assert public.save_dir == paths.noise_model_dir(noiseModel_name=public.name)
    assert public.n_pairs == 3
    assert public.signal_shape == legacy.signal_shape == (3, 10, 12)
    assert public.observation_shape == legacy.observation_shape == (3, 10, 12)
    np.testing.assert_array_equal(signal, input_signal)
    np.testing.assert_array_equal(observation, input_observation)
    np.testing.assert_array_equal(np.load(public.histogram_path), np.load(legacy.histogram_path))
    assert json.loads(public.norm_prm_path.read_text()) == json.loads(legacy.norm_prm_path.read_text())
    with np.load(public.gmm_path) as public_npz, np.load(legacy.gmm_path) as legacy_npz:
        for key in ("trained_weight", "min_signal", "max_signal", "min_sigma"):
            np.testing.assert_array_equal(public_npz[key], legacy_npz[key])
    model, norms = load_noise_model_from_paths(
        public.gmm_path, public.norm_prm_path, torch.device("cpu")
    )
    assert model.n_gaussian == 1
    assert norms["normalize_data"] is True
    assert "in-memory paired NumPy arrays" in public.info_path.read_text()
    assert public.gmm_plot_png_path is None
    assert public.gmm_plot_svg_path is None


def test_single_2d_pair_and_workspace_selection(settings, sample_pairs, options, tmp_path):
    sig, obs = (pair[0].copy() for pair in sample_pairs)
    first = _create(settings, (sig, obs), options, name="first")
    assert first.n_pairs == 1
    assert first.signal_shape == first.observation_shape == (1, 10, 12)
    assert first.gmm_path.is_file()
    second_workspace = tmp_path / "other"
    initialize_workspace(second_workspace)
    second = _create(Settings.from_workspace(second_workspace), (sig, obs), options, name="first")
    assert second.save_dir != first.save_dir
    assert second.save_dir.parent == second_workspace / "noise_models"


def test_explicit_overwrite_and_no_auto_numbering(settings, sample_pairs, options):
    first = _create(settings, sample_pairs, options, name="recalibrated")
    with pytest.raises(FileExistsError, match="already exists"):
        _create(settings, sample_pairs, options, name="recalibrated")
    second = _create(settings, sample_pairs, options, name="recalibrated", overwrite=True)
    assert second.save_dir == first.save_dir
    assert second.gmm_path.exists()
    assert not first.save_dir.with_name("recalibrated_0").exists()


@pytest.mark.parametrize("name", ("", "../escape", "foo/bar", r"foo\bar", "../", ".", "..", "bad name"))
def test_rejects_unsafe_name_before_writing(settings, sample_pairs, options, name):
    with pytest.raises(ValueError, match="name must"):
        _create(settings, sample_pairs, options, name=name)
    assert not any((Paths(settings).workspace_root() / "noise_models").iterdir())


@pytest.mark.parametrize("modification, error", [
    (lambda s, o: (s[:, :, :7], o), "shapes must match"),
    (lambda s, o: (s[0, 0], o[0, 0]), "shape"),
    (lambda s, o: (s[:, None], o[:, None]), "shape"),
    (lambda s, o: (s.astype(complex), o), "real numeric"),
    (lambda s, o: (s.astype(bool), o), "real numeric"),
    (lambda s, o: (s[:0], o[:0]), "empty"),
    (lambda s, o: (np.full_like(s, np.nan), o), "non-finite"),
    (lambda s, o: (np.ones_like(s), o), "standard deviation"),
])
def test_bad_array_does_not_create_calibration(
    settings, sample_pairs, options, modification, error,
):
    signal, observation = modification(*sample_pairs)
    with pytest.raises((TypeError, ValueError), match=error):
        _create(settings, (signal, observation), options)
    assert not Paths(settings).noise_model_dir(noiseModel_name="calibrated").exists()


def test_bad_numpy_type_and_bad_crop_are_rejected(settings, sample_pairs, options):
    with pytest.raises(TypeError, match="NumPy ndarray"):
        _create(settings, (sample_pairs[0].tolist(), sample_pairs[1]), options)
    invalid_crop = NoiseModelOptions(
        histogram_bins=16, gmm_n_gaussian=1, gmm_n_coeff=2, gmm_n_epochs=4,
        crop_size=(100, 100), save_gmm_plot=False,
    )
    with pytest.raises(ValueError, match="exceeds image size"):
        _create(settings, sample_pairs, invalid_crop)
    assert not Paths(settings).noise_model_dir(noiseModel_name="calibrated").exists()


@pytest.mark.parametrize("bad", [
    {"gmm_n_epochs": 1}, {"gmm_n_coeff": 1}, {"histogram_bins": 0},
    {"gmm_learning_rate": float("nan")}, {"gmm_min_sigma": -1},
    {"crop_size": (2, 0)}, {"norm_sig_to_obs": "yes"},
])
def test_invalid_options_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        NoiseModelOptions(**bad)


def test_explicit_cuda_unavailable_fails_before_write(settings, sample_pairs, options, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        create_noise_model(*sample_pairs, name="calibrated", settings=settings,
                           options=options, device="cuda")
    assert not Paths(settings).noise_model_dir(noiseModel_name="calibrated").exists()


def test_optional_plot_saved_with_current_backend(settings, sample_pairs, options):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    plotted = NoiseModelOptions(
        histogram_bins=16, gmm_n_gaussian=1, gmm_n_coeff=2, gmm_n_epochs=4,
        gmm_batch_size=100, gmm_min_sigma=0.1, gmm_plot_points=4,
        save_gmm_plot=True,
    )
    result = _create(settings, sample_pairs, plotted)
    assert result.gmm_plot_png_path is not None and result.gmm_plot_png_path.is_file()
    assert result.gmm_plot_svg_path is not None and result.gmm_plot_svg_path.is_file()


def test_new_model_works_with_quick_hdn_preflight(settings, sample_pairs, options):
    from lisai.data.import_training import import_training_dataset
    from lisai.training.quick_hdn import prepare_quick_hdn_training

    calibration = _create(settings, sample_pairs, options, name="my_camera")
    data = [np.ones((3, 64, 64), dtype=np.uint16) * (i + 1) for i in range(2)]
    import_training_dataset(data, settings=settings, dataset_name="mito", data_format="timelapse")
    plan = prepare_quick_hdn_training(
        settings=settings,
        dataset_name="mito",
        noise_model_name=calibration.name,
        experiment_name="hdn_first",
        epochs=2,
        patch_size=32,
    )
    assert plan.noise_model_name == calibration.name
    assert plan.config["noise_model"]["name"] == calibration.name
    assert plan.config["normalization"]["load_from_noise_model"] is True


def test_failed_fit_cleans_only_new_model_directory(settings, sample_pairs, options, monkeypatch):
    import lisai.noise_models as public_module

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic fitting failure")

    monkeypatch.setattr(public_module, "_fit_and_save_prepared", fail)
    with pytest.raises(RuntimeError, match="synthetic fitting failure"):
        _create(settings, sample_pairs, options)
    assert not Paths(settings).noise_model_dir(noiseModel_name="calibrated").exists()
