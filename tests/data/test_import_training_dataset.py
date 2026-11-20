"""L1: independent raw-recording import into a selected LISAI workspace."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.data.dataset_registry import load_dataset_registry
from lisai.data.import_training import import_training_dataset
from lisai.infra.paths import Paths


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "ws"
    initialize_workspace(root)
    return Settings.from_workspace(root)


def test_import_timelapses_keeps_frames_raw_dtype_and_registry(settings, tmp_path):
    raw = np.arange(4 * 32 * 36, dtype=np.uint16).reshape(4, 32, 36)
    disk = tmp_path / "source image.TIFF"
    tifffile.imwrite(disk, raw)
    second = np.ones((3, 32, 36), dtype=np.float32)
    result = import_training_dataset(
        [disk, ("next_timelapse.tif", second)], settings=settings,
        dataset_name="mito_hdn", data_format="timelapse",
    )
    assert result.recording_count == 2
    assert result.recording_paths[0].name == "source image.tiff"
    assert result.total_frames == 7
    assert result.dataset_dir == Paths(settings).dataset_dir(dataset_name="mito_hdn")
    np.testing.assert_array_equal(tifffile.imread(result.recording_paths[0]), raw)
    np.testing.assert_array_equal(tifffile.imread(result.recording_paths[1]), second)
    info = load_dataset_registry(Paths(settings).dataset_registry_path())["mito_hdn"]
    assert info["data_format"] == "timelapse"
    assert info["defaults"]["raw"]["input"] == "raw"
    assert info["outputs"]["raw"][0]["axes"] == "TYX"


def test_single_arrays_and_workspace_isolation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        initialize_workspace(root)
    image = np.full((48, 48), 33, np.uint8)
    import_training_dataset([image, image], settings=Settings.from_workspace(a),
                            dataset_name="cells", data_format="single")
    assert not Paths(Settings.from_workspace(b)).dataset_dir(dataset_name="cells").exists()
    assert len(load_dataset_registry(Paths(Settings.from_workspace(a)).dataset_registry_path())) == 1


def test_reject_collision_and_unsafe_inputs(settings, tmp_path):
    original = np.ones((2, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="collide"):
        import_training_dataset([("a.TIF", original), ("A.tif", original)],
                                settings=settings, dataset_name="bad", data_format="timelapse")
    with pytest.raises(ValueError, match="dataset name"):
        import_training_dataset([original], settings=settings, dataset_name="../other", data_format="timelapse")
    with pytest.raises(ValueError, match="expected TYX"):
        import_training_dataset([original[0]], settings=settings,
                                dataset_name="bad", data_format="timelapse")
    with pytest.raises(ValueError, match="NaN"):
        import_training_dataset([np.full((4, 4), np.nan)], settings=settings,
                                dataset_name="bad", data_format="single")
    assert not Paths(settings).dataset_dir(dataset_name="bad").exists()
    assert load_dataset_registry(Paths(settings).dataset_registry_path()) == {}


def test_no_overwrite_and_registry_rollback(settings, monkeypatch):
    import lisai.data.import_training as module
    arr = np.ones((32, 32), np.uint16)
    import_training_dataset([arr], settings=settings, dataset_name="one", data_format="single")
    with pytest.raises(FileExistsError):
        import_training_dataset([arr], settings=settings, dataset_name="one", data_format="single")
    before = Paths(settings).dataset_registry_path().read_bytes()
    def broken_replace(*args):
        raise OSError("forced registry failure")
    monkeypatch.setattr(module.os, "replace", broken_replace)
    with pytest.raises(OSError, match="forced"):
        import_training_dataset([arr], settings=settings, dataset_name="two", data_format="single")
    assert Paths(settings).dataset_registry_path().read_bytes() == before
    assert not Paths(settings).dataset_dir(dataset_name="two").exists()
