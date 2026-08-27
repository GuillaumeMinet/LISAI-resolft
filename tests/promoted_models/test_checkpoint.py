from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.models.params import UNetParams
from lisai.promoted_models.checkpoint import extract_model_weights, resolve_promotion_checkpoint


def _saved_run(tmp_path: Path, *, methods=("state_dict",)) -> SavedTrainingRun:
    return SavedTrainingRun(
        run_dir=tmp_path,
        experiment_name="run_a",
        dataset_name="dataset_a",
        data_subfolder="",
        data_cfg={},
        model_architecture="unet",
        model_parameters=UNetParams(),
        data_norm_prm=None,
        model_norm_prm=None,
        noise_model_name=None,
        checkpoint_methods=methods,
        patch_size=64,
        downsamp_factor=1,
        upsampling_factor=1,
        context_length=None,
        default_tiling_size=1024,
    )


class FakePaths:
    def checkpoint_path(self, *, run_dir, load_method, best_or_last):
        return Path(run_dir) / "checkpoints" / f"model_{best_or_last}_{load_method}.pt"


def test_resolve_promotion_checkpoint_uses_canonical_state_dict(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "model_best_state_dict.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_promotion_checkpoint(_saved_run(tmp_path), paths=FakePaths())

    assert resolved == checkpoint


def test_resolve_promotion_checkpoint_requires_state_dict_support(tmp_path: Path):
    with pytest.raises(ValueError, match="requires a state-dict checkpoint"):
        resolve_promotion_checkpoint(_saved_run(tmp_path, methods=("full_model",)), paths=FakePaths())


def test_extract_model_weights_strips_training_state(tmp_path: Path):
    source = tmp_path / "checkpoint.pt"
    output = tmp_path / "weights.pt"
    expected = {"layer.weight": torch.tensor([1.0, 2.0])}
    torch.save(
        {
            "epoch": 4,
            "optimizer_state_dict": {"state": {}},
            "model_state_dict": expected,
        },
        source,
    )

    extract_model_weights(source, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)

    assert set(loaded) == {"layer.weight"}
    assert torch.equal(loaded["layer.weight"], expected["layer.weight"])


def test_extract_model_weights_accepts_plain_state_dict(tmp_path: Path):
    source = tmp_path / "checkpoint.pt"
    output = tmp_path / "weights.pt"
    torch.save({"layer.bias": torch.tensor([3.0])}, source)

    extract_model_weights(source, output)

    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.equal(loaded["layer.bias"], torch.tensor([3.0]))


def test_extract_model_weights_rejects_arbitrary_training_dict(tmp_path: Path):
    source = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 2, "optimizer_state_dict": {}}, source)

    with pytest.raises(ValueError, match="does not contain model_state_dict"):
        extract_model_weights(source, tmp_path / "weights.pt")
