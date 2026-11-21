"""Patch C: all-or-nothing persistence for both calibration entry points."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.data.noise_model import NoiseModelBuildConfig, build_noise_model
from lisai.data.noise_model.io import staged_noise_model_output
from lisai.infra.paths import Paths
from lisai.models.load_nm import load_noise_model_from_paths
from lisai.noise_models import NoiseModelOptions, create_noise_model


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "workspace"
    initialize_workspace(root)
    return Settings.from_workspace(root)


@pytest.fixture
def images():
    yy, xx = np.mgrid[:12, :14]
    signal = np.stack([
        2 + 0.5 * xx + 0.3 * yy + np.sin(xx + yy + i) + i for i in range(3)
    ]).astype(np.float32)
    return signal, (signal * .8 + 2 + np.cos(yy * 1.8)).astype(np.float32)


@pytest.fixture
def options():
    return NoiseModelOptions(
        histogram_bins=16, gmm_n_gaussian=1, gmm_n_coeff=2,
        gmm_n_epochs=4, gmm_batch_size=100, gmm_min_sigma=.1,
        save_gmm_plot=False,
    )


def _create(settings, images, options, **kwargs):
    return create_noise_model(
        *images, settings=settings, name="camera", options=options, device="cpu", **kwargs
    )


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


def _no_temporary_dirs(folder: Path) -> None:
    assert not [
        p.name for p in folder.iterdir()
        if p.name.startswith(".camera.staging-") or p.name.startswith(".camera.backup-")
    ]


def test_failed_new_calibration_never_exposes_partial_files(settings, images, options, monkeypatch):
    import lisai.data.noise_model.builder as builder

    def fail_after_initial_writes(*args, **kwargs):
        work_dir = kwargs["save_dir"]
        (work_dir / "Histogram.npy").write_bytes(b"partial")
        raise RuntimeError("fit interrupted")

    monkeypatch.setattr(builder, "_fit_and_save_prepared", fail_after_initial_writes)
    with pytest.raises(RuntimeError, match="fit interrupted"):
        _create(settings, images, options)
    directory = Paths(settings).noise_model_dir(noiseModel_name="camera")
    assert not directory.exists()
    _no_temporary_dirs(directory.parent)


def test_failed_overwrite_preserves_old_model_byte_for_byte(settings, images, options, monkeypatch):
    import lisai.data.noise_model.builder as builder

    first = _create(settings, images, options)
    (first.save_dir / "user_note.txt").write_text("important calibration notes")
    old = _snapshot(first.save_dir)

    def fail_after_initial_writes(*args, **kwargs):
        work_dir = kwargs["save_dir"]
        (work_dir / "GMM.npz").write_bytes(b"incomplete")
        raise RuntimeError("fit interrupted")

    monkeypatch.setattr(builder, "_fit_and_save_prepared", fail_after_initial_writes)
    with pytest.raises(RuntimeError, match="fit interrupted"):
        _create(settings, images, options, overwrite=True)
    assert _snapshot(first.save_dir) == old
    _no_temporary_dirs(first.save_dir.parent)


def test_successful_overwrite_replaces_whole_directory_and_removes_stale_artifacts(
    settings, images, options,
):
    first = _create(settings, images, options)
    (first.save_dir / "obsolete_GMM_plot.png").write_bytes(b"old plot")
    (first.save_dir / "obsolete.npz").write_bytes(b"old model")
    next_images = (images[0] + 2, images[1] + 5)
    second = _create(settings, next_images, options, overwrite=True)
    assert second.save_dir == first.save_dir
    assert set(_snapshot(second.save_dir)) == {
        "Histogram.npy", "GMM.npz", "norm_prm.json", "info.txt"
    }
    model, norm = load_noise_model_from_paths(
        second.gmm_path, second.norm_prm_path, torch.device("cpu")
    )
    assert model.n_gaussian == 1
    assert norm == json.loads(second.norm_prm_path.read_text())
    _no_temporary_dirs(second.save_dir.parent)


def test_publish_rename_failure_restores_old_calibration(settings, images, options, monkeypatch):
    first = _create(settings, images, options)
    old = _snapshot(first.save_dir)
    original_rename = Path.rename

    def fail_publish(self, target):
        if self.name.startswith(".camera.staging-") and Path(target) == first.save_dir:
            raise OSError("synthetic publishing error")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="synthetic publishing error"):
        _create(settings, images, options, overwrite=True)
    assert _snapshot(first.save_dir) == old
    _no_temporary_dirs(first.save_dir.parent)


def test_legacy_builder_collision_suffix_and_failed_model_cleanup(settings, images):
    paths = Paths(settings)
    root = paths.dataset_dir(dataset_name="paired")
    (root / "sig").mkdir(parents=True)
    (root / "obs").mkdir()
    for i, (sig, obs) in enumerate(zip(*images)):
        tifffile.imwrite(root / "sig" / f"{i}.tif", sig)
        tifffile.imwrite(root / "obs" / f"{i}.tif", obs)
    cfg = NoiseModelBuildConfig(
        dataset_name="paired", signal_subfolder="sig", observation_subfolder="obs",
        save_name="camera", histogram_bins=16, gmm_n_gaussian=1,
        gmm_n_coeff=2, gmm_n_epochs=4, gmm_batch_size=100,
        gmm_min_sigma=.1, save_gmm_plot=False,
    )
    first = build_noise_model(cfg, paths=paths, device=torch.device("cpu"))
    second = build_noise_model(cfg, paths=paths, device=torch.device("cpu"))
    assert first.save_dir.name == "camera"
    assert second.save_dir.name == "camera_0"
    assert first.gmm_path.is_file() and second.gmm_path.is_file()
    _no_temporary_dirs(first.save_dir.parent)


def test_plot_dependency_failure_isolated_to_plot_request(settings, images, options, monkeypatch):
    import lisai.data.noise_model.plotting as plotting

    def missing_matplotlib():
        raise RuntimeError("Noise-model plotting requires Matplotlib")

    monkeypatch.setattr(plotting, "_pyplot", missing_matplotlib)
    # matplotlib must not be needed for non-plotting fits.
    result = _create(settings, images, options)
    assert result.gmm_path.exists()
    plot_options = NoiseModelOptions(
        histogram_bins=16, gmm_n_gaussian=1, gmm_n_coeff=2,
        gmm_n_epochs=4, gmm_batch_size=100, gmm_min_sigma=.1, save_gmm_plot=True,
    )
    old = _snapshot(result.save_dir)
    with pytest.raises(RuntimeError, match="Matplotlib"):
        _create(settings, images, plot_options, overwrite=True)
    assert _snapshot(result.save_dir) == old
    _no_temporary_dirs(result.save_dir.parent)


def test_no_publish_when_fitting_returns_missing_artifacts(settings, images, options, monkeypatch):
    import lisai.data.noise_model.builder as builder

    original = builder._fit_and_save_prepared

    def remove_artifact(*args, **kwargs):
        built = original(*args, **kwargs)
        built.norm_prm_path.unlink()
        return built

    monkeypatch.setattr(builder, "_fit_and_save_prepared", remove_artifact)
    with pytest.raises(RuntimeError, match="valid artifact"):
        _create(settings, images, options)
    assert not Paths(settings).noise_model_dir(noiseModel_name="camera").exists()
    _no_temporary_dirs(Paths(settings).workspace_root() / "noise_models")
