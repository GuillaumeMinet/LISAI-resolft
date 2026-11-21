"""L2: workspace-bound quick unsupervised HDN configuration creation."""
from __future__ import annotations

import json

import numpy as np
import pytest

from lisai.config.io.resolver import resolve_config
from lisai.config.models.training.data import SplitRatios
from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.data.import_training import import_training_dataset
from lisai.infra.paths import Paths
from lisai.training.quick_hdn import prepare_quick_hdn_training


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "workspace"
    initialize_workspace(root)
    settings = Settings.from_workspace(root)
    noise_dir = Paths(settings).noise_model_dir(noiseModel_name="calibrated")
    noise_dir.mkdir(parents=True)
    np.savez(noise_dir / "GMM.npz", min_signal=np.array([0.0]), max_signal=np.array([255.0]),
             trained_weight=np.zeros((3, 2)), min_sigma=np.array([1.0]))
    (noise_dir / "norm_prm.json").write_text(json.dumps({"clip": 0, "normalize_data": False}), encoding="utf-8")
    return settings


def _add(settings, n=2, fmt="timelapse", name="mito"):
    images = ([np.ones((4, 64, 64), dtype=np.uint16) * (index + 1) for index in range(n)]
              if fmt == "timelapse" else
              [np.ones((64, 64), dtype=np.float32) * (index + 1) for index in range(n)])
    return import_training_dataset(images, settings=settings, dataset_name=name, data_format=fmt)


def _plan(settings, **kwargs):
    return prepare_quick_hdn_training(
        settings=settings, dataset_name=kwargs.pop("dataset_name", "mito"),
        noise_model_name=kwargs.pop("noise_model_name", "calibrated"),
        experiment_name=kwargs.pop("experiment_name", "first_hdn"),
        epochs=kwargs.pop("epochs", 2), patch_size=kwargs.pop("patch_size", 32), **kwargs,
    )


def test_two_timelapses_preflight_and_saved_resolved_config(setup):
    _add(setup)
    plan = _plan(setup, seed=7)
    assert plan.split_counts == {"train": 1, "val": 1, "test": 0}
    assert not plan.config_path.exists()
    assert plan.config["data"]["timelapse_prm"]["context_length"] is None
    saved = _plan(setup, seed=7, write=True)
    assert saved.config_path.is_file()
    cfg = resolve_config(saved.config_path, stg=setup)
    assert cfg.data.prep_before is False
    assert cfg.data.timelapse_prm is not None and cfg.data.timelapse_prm.context_length is None
    assert cfg.model.architecture == "lvae"
    assert cfg.experiment.post_training_inference is False
    assert cfg.training.n_epochs == 2
    assert cfg.data.split_manifest.ratios.test == 0
    assert cfg.routing.data_subfolder == ""
    with pytest.raises(FileExistsError, match="config already"):
        _plan(setup, write=True)


def test_one_file_rejected_without_config_or_run(setup):
    _add(setup, n=1)
    with pytest.raises(ValueError, match="two separate recordings"):
        _plan(setup, write=True)
    assert not (setup.TRAINING_CONFIG_DIR / "improcess" / "first_hdn.yml").exists()
    assert not Paths(setup).dataset_runs_dir(dataset_name="mito").exists()


def test_three_files_optional_heldout_test_and_single_images(setup):
    _add(setup, n=3, fmt="single")
    plan = _plan(setup, split_ratios=SplitRatios(train=0.6, val=0.2, test=0.2))
    assert plan.split_counts == {"train": 1, "val": 1, "test": 1}
    assert "timelapse_prm" not in plan.config["data"]
    assert set(sum([list(group) for group in plan.split_filenames.values()], [])) == {
        "recording_0001.tif", "recording_0002.tif", "recording_0003.tif"
    }


def test_workspace_isolation_invalid_noise_and_invalid_split(setup, tmp_path):
    _add(setup)
    with pytest.raises(FileNotFoundError, match="noise model"):
        _plan(setup, noise_model_name="absent")
    with pytest.raises(ValueError, match="at least 3 files"):
        _plan(setup, split_ratios={"train": .6, "val": .2, "test": .2})
    other = tmp_path / "other_workspace"
    initialize_workspace(other)
    with pytest.raises(ValueError, match="not registered"):
        _plan(Settings.from_workspace(other))


def test_reject_too_small_images_and_invalid_parameters(setup):
    _add(setup)
    with pytest.raises(ValueError, match="Y/X"):
        _plan(setup, patch_size=512)
    with pytest.raises(ValueError, match="epochs"):
        _plan(setup, epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        _plan(setup, learning_rate=float("nan"))


def test_single_frame_tyx_is_still_one_timelapse_recording(setup):
    import_training_dataset(
        [np.ones((1, 64, 64), dtype=np.uint16), np.ones((2, 64, 64), dtype=np.uint16)],
        settings=setup, dataset_name="mito", data_format="timelapse",
    )
    plan = _plan(setup)
    assert plan.split_counts == {"train": 1, "val": 1, "test": 0}


def test_invalid_noise_model_archive_rejected_before_writing(setup):
    _add(setup)
    nm = Paths(setup).noise_model_path(noiseModel_name="calibrated")
    np.savez(nm, trained_weight=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="missing fields"):
        _plan(setup, write=True)
    assert not (setup.TRAINING_CONFIG_DIR / "improcess" / "first_hdn.yml").exists()


def test_quick_hdn_scientific_beta_override_uses_task_schema(setup):
    _add(setup)
    plan = _plan(setup, beta_kl=0.25)
    assert plan.config["experiment"]["task"]["betaKL"] == 0.25
    with pytest.raises(ValueError, match="beta_kl"):
        _plan(setup, beta_kl=-0.2)
