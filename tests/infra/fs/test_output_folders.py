from __future__ import annotations

from pathlib import Path

import pytest

from lisai.infra.fs import OutputFolderResolution, prepare_output_folder


def test_prepare_output_folder_creates_missing_folder(tmp_path: Path):
    requested = tmp_path / "predictions"

    resolution = prepare_output_folder(requested, if_exists_policy="numbered")

    assert resolution == OutputFolderResolution(
        requested=requested,
        path=requested,
        action="created",
        requested_existed=False,
    )
    assert requested.is_dir()
    assert resolution.message() == f"SAVING: Saving outputs to: {requested}"


def test_prepare_output_folder_numbers_existing_destination(tmp_path: Path):
    requested = tmp_path / "source" / "model"
    requested.mkdir(parents=True)

    resolution = prepare_output_folder(requested, if_exists_policy="numbered")

    assert resolution.requested == requested
    assert resolution.path == tmp_path / "source" / "model_01"
    assert resolution.action == "numbered"
    assert resolution.requested_existed is True
    assert resolution.redirected is True
    assert resolution.path.is_dir()
    assert resolution.message() == (
        f"SAVING: Folder {requested} already exists; "
        f"saving to {resolution.path} instead. "
        "Use --overwrite to replace the existing folder."
    )


def test_prepare_output_folder_overwrites_existing_destination(tmp_path: Path):
    requested = tmp_path / "source" / "model"
    requested.mkdir(parents=True)
    old_file = requested / "old_prediction.tif"
    old_file.write_text("old")

    resolution = prepare_output_folder(requested, if_exists_policy="overwrite")

    assert resolution.path == requested
    assert resolution.action == "overwritten"
    assert requested.is_dir()
    assert not old_file.exists()
    assert resolution.message() == (
        f"SAVING: Folder {requested} already exists; --overwrite enabled, replacing it."
    )


def test_prepare_output_folder_reuses_existing_destination(tmp_path: Path):
    requested = tmp_path / "predictions"
    requested.mkdir()
    old_file = requested / "old_prediction.tif"
    old_file.write_text("old")

    resolution = prepare_output_folder(requested, if_exists_policy="reuse")

    assert resolution.path == requested
    assert resolution.action == "reused"
    assert old_file.exists()
    assert resolution.message() == f"SAVING: Reusing output folder: {requested}"


def test_prepare_output_folder_reuse_creates_missing_destination(tmp_path: Path):
    requested = tmp_path / "predictions"

    resolution = prepare_output_folder(requested, if_exists_policy="reuse")

    assert resolution.path == requested
    assert resolution.action == "created"
    assert requested.is_dir()
    assert resolution.message() == f"SAVING: Saving outputs to: {requested}"


def test_prepare_output_folder_error_policy_refuses_existing_destination(tmp_path: Path):
    requested = tmp_path / "predictions"
    requested.mkdir()

    with pytest.raises(FileExistsError):
        prepare_output_folder(requested, if_exists_policy="error")


def test_prepare_output_folder_requires_existing_parent(tmp_path: Path):
    requested = tmp_path / "missing" / "predictions"

    with pytest.raises(FileNotFoundError, match="Parent folder does not exist"):
        prepare_output_folder(
            requested,
            if_exists_policy="numbered",
            parent_policy="require",
        )
