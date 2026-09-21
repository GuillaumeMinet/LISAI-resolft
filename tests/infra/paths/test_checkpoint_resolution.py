from __future__ import annotations

from pathlib import Path

import pytest

from lisai.infra.paths.checkpoint_resolution import resolve_checkpoint_path


class FakePaths:
    def checkpoint_path(
        self,
        *,
        run_dir,
        load_method=None,
        best_or_last=None,
        epoch_number=None,
    ):
        if epoch_number is not None:
            middle = f"epoch_{epoch_number}"
        else:
            middle = best_or_last

        if load_method == "state_dict":
            filename = f"model_{middle}_state_dict.pt"
        elif load_method == "full_model":
            filename = f"model_{middle}.pt"
        else:
            raise ValueError(f"Unknown load_method: {load_method}")

        return Path(run_dir) / "checkpoints" / filename


def test_resolve_checkpoint_path_prefers_canonical_best_over_epoch_files(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    canonical_best = checkpoints / "model_best_state_dict.pt"
    epoch_best = checkpoints / "model_epoch_9_state_dict.pt"
    canonical_best.write_bytes(b"best")
    epoch_best.write_bytes(b"epoch")

    method, path = resolve_checkpoint_path(
        paths=FakePaths(),
        run_dir=tmp_path,
        load_methods=("state_dict",),
        best_or_last="best",
    )

    assert method == "state_dict"
    assert path == canonical_best


def test_resolve_checkpoint_path_falls_back_to_highest_epoch_best(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "model_epoch_2_state_dict.pt").write_bytes(b"epoch-2")
    highest = checkpoints / "model_epoch_7_state_dict.pt"
    highest.write_bytes(b"epoch-7")

    method, path = resolve_checkpoint_path(
        paths=FakePaths(),
        run_dir=tmp_path,
        load_methods=("state_dict",),
        best_or_last="best",
    )

    assert method == "state_dict"
    assert path == highest


def test_resolve_checkpoint_path_full_model_ignores_state_dict_epoch_files(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "model_epoch_99_state_dict.pt").write_bytes(b"state")
    full_model = checkpoints / "model_epoch_3.pt"
    full_model.write_bytes(b"full")

    method, path = resolve_checkpoint_path(
        paths=FakePaths(),
        run_dir=tmp_path,
        load_methods=("full_model",),
        best_or_last="best",
    )

    assert method == "full_model"
    assert path == full_model


def test_resolve_checkpoint_path_does_not_use_epoch_files_for_last(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "model_epoch_7_state_dict.pt").write_bytes(b"epoch")

    with pytest.raises(FileNotFoundError, match="Could not find model checkpoint"):
        resolve_checkpoint_path(
            paths=FakePaths(),
            run_dir=tmp_path,
            load_methods=("state_dict",),
            best_or_last="last",
        )
