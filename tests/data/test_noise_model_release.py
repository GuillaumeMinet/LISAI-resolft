"""Release-facing smoke coverage for the NumPy noise-model calibration API."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.infra.paths import Paths
from lisai.models.load_nm import load_noise_model
from lisai.noise_models import NoiseModelOptions, create_noise_model


def _pairs() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:16, :18]
    signal = np.stack([
        10.0 + 0.45 * xx + 0.65 * yy + np.sin(xx + yy + i) + i
        for i in range(3)
    ]).astype(np.float32)
    observation = (signal * 0.75 + 4.0 + np.cos(yy * 1.1)).astype(np.float32)
    return signal, observation


def _options() -> NoiseModelOptions:
    return NoiseModelOptions(
        histogram_bins=16,
        gmm_n_gaussian=1,
        gmm_n_coeff=2,
        gmm_n_epochs=4,
        gmm_batch_size=100,
        gmm_min_sigma=0.1,
        save_gmm_plot=False,
    )


def test_saved_calibration_loads_by_name_and_evaluates_finite_likelihood(tmp_path):
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    settings = Settings.from_workspace(workspace)
    paths = Paths(settings)
    signal, observation = _pairs()
    original_signal, original_observation = signal.copy(), observation.copy()

    np.random.seed(29)
    torch.manual_seed(29)
    result = create_noise_model(
        signal, observation, name="camera_488", settings=settings,
        options=_options(), device="cpu",
    )

    assert result.name == "camera_488"
    assert result.save_dir == paths.noise_model_dir(noiseModel_name="camera_488")
    assert result.n_pairs == 3
    assert {p.name for p in result.save_dir.iterdir()} == {
        "Histogram.npy", "GMM.npz", "norm_prm.json", "info.txt"
    }
    assert result.histogram_path.stat().st_size > 0
    np.testing.assert_array_equal(signal, original_signal)
    np.testing.assert_array_equal(observation, original_observation)

    norms = json.loads(result.norm_prm_path.read_text(encoding="utf-8"))
    assert norms["normalize_data"] is True
    assert norms["normSig2Obs"] is True
    assert all(np.isfinite(norms[k]) for k in ("avgSig", "stdSig", "avgObs", "stdObs"))
    histogram = np.load(result.histogram_path)
    assert histogram.shape == (3, 16, 16)
    assert np.isfinite(histogram).all()

    # Load through the same name-based entry point as HDN training.
    loaded, loaded_norms = load_noise_model("camera_488", torch.device("cpu"), paths)
    assert loaded_norms == norms
    assert loaded.n_gaussian == 1
    assert loaded.n_coeff == 2
    # Evaluate P(observed | signal) in the normalized fitting domain.
    source = torch.linspace(float(loaded.min_signal.item()) + 0.1,
                            float(loaded.max_signal.item()) - 0.1, 16)
    with torch.no_grad():
        likelihood = loaded.likelihood(source, source)
    assert likelihood.shape == source.shape
    assert torch.isfinite(likelihood).all()
    assert (likelihood > 0).all()


def test_import_and_fit_without_optional_plotting_dependencies(tmp_path):
    """The no-plot path must work with base dependencies, without sklearn/Matplotlib.

    Use a fresh interpreter to avoid misleading passes due to already-imported
    optional libraries in the parent test process.
    """
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    code = r'''
import builtins
import numpy as np

real_import = builtins.__import__
def block_optional(name, *args, **kwargs):
    if name.split(".", 1)[0] in ("matplotlib", "sklearn"):
        raise ImportError("Optional dependency disabled for calibration smoke test: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = block_optional

from lisai.config.settings import Settings
from lisai.noise_models import NoiseModelOptions, create_noise_model

settings = Settings.from_workspace(WS)
y, x = np.mgrid[:12, :14]
signal = (5 + x * .6 + y * .4 + np.sin(x + y)).astype(np.float32)
observation = (signal * .8 + 2 + np.cos(y)).astype(np.float32)
result = create_noise_model(
    signal, observation, name="without_plot", settings=settings,
    device="cpu", options=NoiseModelOptions(
        histogram_bins=16, gmm_n_gaussian=1, gmm_n_coeff=2,
        gmm_n_epochs=4, gmm_batch_size=100, gmm_min_sigma=.1,
        save_gmm_plot=False,
    ),
)
assert result.gmm_path.is_file()
assert result.norm_prm_path.is_file()
assert result.gmm_plot_png_path is None
'''.replace("WS", repr(str(workspace)))
    env = dict(os.environ)
    # Prevent source-tree-dependent imports inside subprocesses when tests are
    # launched from a different directory and the package is installed editable.
    env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=tmp_path, env=env, timeout=90, check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    assert (workspace / "noise_models" / "without_plot" / "GMM.npz").exists()
