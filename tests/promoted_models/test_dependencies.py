from __future__ import annotations

from pathlib import Path

import pytest

from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.models.params import LVAEParams, UNetParams
from lisai.promoted_models.dependencies import collect_required_dependencies


def _saved_run(
    tmp_path: Path,
    *,
    architecture: str,
    noise_model_name: str | None = None,
) -> SavedTrainingRun:
    params = LVAEParams(num_latents=2, z_dims=16) if architecture == "lvae" else UNetParams()
    return SavedTrainingRun(
        run_dir=tmp_path,
        experiment_name="run_a",
        dataset_name="dataset_a",
        data_subfolder="",
        data_cfg={},
        model_architecture=architecture,
        model_parameters=params,
        data_norm_prm=None,
        model_norm_prm=None,
        noise_model_name=noise_model_name,
        checkpoint_methods=("state_dict",),
        patch_size=64,
        downsamp_factor=1,
        upsampling_factor=1,
        context_length=None,
        default_tiling_size=300 if architecture == "lvae" else 1024,
    )


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def noise_model_path(self, *, noiseModel_name: str):
        return self.root / noiseModel_name / "GMM.npz"

    def noise_model_norm_prm_path(self, *, noiseModel_name: str):
        return self.root / noiseModel_name / "norm_prm.json"


def test_non_lvae_has_no_external_dependencies(tmp_path: Path):
    deps = collect_required_dependencies(
        _saved_run(tmp_path, architecture="unet"),
        paths=FakePaths(tmp_path / "noise_models"),
    )

    assert deps.files == ()
    assert deps.noise_model is None


def test_lvae_bundles_canonical_noise_model_files(tmp_path: Path):
    noise_dir = tmp_path / "noise_models" / "NoiseA"
    noise_dir.mkdir(parents=True)
    (noise_dir / "GMM.npz").write_bytes(b"gmm")
    (noise_dir / "norm_prm.json").write_text("{}", encoding="utf-8")

    deps = collect_required_dependencies(
        _saved_run(tmp_path, architecture="lvae", noise_model_name="NoiseA"),
        paths=FakePaths(tmp_path / "noise_models"),
    )

    assert [item.package_path for item in deps.files] == [
        "artifacts/noise_model/GMM.npz",
        "artifacts/noise_model/norm_prm.json",
    ]
    assert deps.noise_model is not None
    assert deps.noise_model.name == "NoiseA"


def test_lvae_promotion_fails_when_noise_model_dependency_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="self-contained noise model"):
        collect_required_dependencies(
            _saved_run(tmp_path, architecture="lvae", noise_model_name="NoiseA"),
            paths=FakePaths(tmp_path / "noise_models"),
        )
