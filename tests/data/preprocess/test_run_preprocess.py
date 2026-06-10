from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lisai.config import load_yaml, settings
from lisai.preprocess import PreprocessRun
from lisai.preprocess.reporting import ConsolePreprocessReporter


class DummyPaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_registry_path(self) -> Path:
        return self.root / "dataset_registry.yml"

    def dataset_dump_dir(self, *, dataset_name: str, data_type: str = "", additional_subfolder: str = "") -> Path:
        return self.root / dataset_name / "dump" / data_type / additional_subfolder

    def dataset_preprocess_dir(self, *, dataset_name: str, data_type: str = "") -> Path:
        return self.root / dataset_name / "preprocess" / data_type

    def preprocess_log_path(self, *, dataset_name: str, data_type: str) -> Path:
        key = f"{data_type}_preprocess"
        return self.dataset_preprocess_dir(dataset_name=dataset_name, data_type=data_type) / settings.data_cfg.logs[key]

    def preprocessed_image_full_path(
        self,
        *,
        dataset_name: str,
        fmt: str,
        data_type: str = "",
        additional_subfolder: str = "",
        **kwargs,
    ) -> Path:
        filename = settings.get_data_filename(fmt=fmt, data_type=data_type, **kwargs)
        return self.dataset_preprocess_dir(dataset_name=dataset_name, data_type=data_type) / additional_subfolder / filename


def _write_single_source_dataset(root: Path, dataset_name: str, file_names: list[str]) -> None:
    dump_dir = root / dataset_name / "dump" / "recon"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for index, file_name in enumerate(file_names):
        image = np.full((8, 8), fill_value=index + 1, dtype=np.uint16)
        tifffile.imwrite(dump_dir / file_name, image)


def _write_timelapse_source_dataset(root: Path, dataset_name: str, *, file_name: str, n_timepoints: int) -> None:
    dump_dir = root / dataset_name / "dump" / "recon"
    dump_dir.mkdir(parents=True, exist_ok=True)
    stack = np.full((n_timepoints, 8, 8), fill_value=1, dtype=np.uint16)
    tifffile.imwrite(dump_dir / file_name, stack)


def _write_broken_single_source_dataset(root: Path, dataset_name: str) -> None:
    dump_dir = root / dataset_name / "dump" / "recon"
    dump_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dump_dir / "img_a.tif", np.full((8, 8), fill_value=1, dtype=np.uint16))
    tifffile.imwrite(dump_dir / "img_b.tif", np.full((2, 8, 8), fill_value=2, dtype=np.uint16))


def _single_cfg(dataset_name: str) -> dict:
    return {
        "dataset_name": dataset_name,
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {},
        "log": {"enabled": True},
        "split": {"enabled": False},
    }


def test_preprocess_run_writes_yaml_manifest_and_manual_split(tmp_path: Path):
    dataset_name = "SampleDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif", "img_b.tif", "img_c.tif"])

    cfg = _single_cfg(dataset_name)
    cfg["split"] = {
        "enabled": True,
        "mode": "manual",
        "manual": {
            "match_by": "source_name",
            "val": ["img_b.tif"],
            "test": ["img_c.tif"],
        },
    }

    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()

    assert result.n_files == 3
    assert (tmp_path / dataset_name / "preprocess" / "recon" / "train" / "c00.tif").exists()
    assert (tmp_path / dataset_name / "preprocess" / "recon" / "val" / "c01.tif").exists()
    assert (tmp_path / dataset_name / "preprocess" / "recon" / "test" / "c02.tif").exists()

    log_path = tmp_path / dataset_name / "preprocess" / "recon" / settings.data_cfg.logs["recon_preprocess"]
    manifest = load_yaml(log_path)
    assert manifest["status"] == "success"
    assert manifest["items"][0]["split"] == "train"
    assert manifest["items"][1]["split"] == "val"
    assert manifest["items"][2]["source_name"] == "img_c.tif"
    assert manifest["summary"]["split"]["counts"] == {"train": 1, "val": 1, "test": 1}
    assert manifest["summary"]["split"]["train"] == {
        "count": 1,
        "source_names": ["img_a.tif"],
        "output_names": ["c00.tif"],
    }
    assert manifest["summary"]["split"]["val"] == {
        "count": 1,
        "source_names": ["img_b.tif"],
        "output_names": ["c01.tif"],
    }
    assert manifest["summary"]["split"]["test"] == {
        "count": 1,
        "source_names": ["img_c.tif"],
        "output_names": ["c02.tif"],
    }

    registry = load_yaml(tmp_path / "dataset_registry.yml")
    assert registry[dataset_name]["data_format"] == "single"
    assert "format" not in registry[dataset_name]
    assert registry[dataset_name]["split"]["recon"]["counts"] == {"train": 1, "val": 1, "test": 1}
    assert "train" not in registry[dataset_name]["split"]["recon"]


def test_preprocess_run_reports_progress_and_final_console_summary(tmp_path: Path):
    dataset_name = "ReportedDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif", "img_b.tif", "img_c.tif"])

    cfg = _single_cfg(dataset_name)
    cfg["split"] = {
        "enabled": True,
        "mode": "manual",
        "manual": {
            "match_by": "source_name",
            "val": ["img_b.tif"],
            "test": ["img_c.tif"],
        },
    }

    stream = io.StringIO()
    reporter = ConsolePreprocessReporter(stream=stream)

    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute(reporter=reporter)

    assert result.n_files == 3
    output = stream.getvalue()
    assert "Preprocess source:" in output
    assert "Preprocess target:" in output
    assert "[1/3] img_a.tif -> c00.tif (split=train)" in output
    assert "[2/3] img_b.tif -> c01.tif (split=val)" in output
    assert "[3/3] img_c.tif -> c02.tif (split=test)" in output
    assert "status: success" in output
    assert "files written: 3" in output
    assert "files moved to validation/test: 2" in output
    assert "validation images (1): img_b.tif -> c01.tif" in output
    assert "test images (1): img_c.tif -> c02.tif" in output


def test_preprocess_run_progress_uses_full_timelapse_output_name(tmp_path: Path):
    dataset_name = "TimelapseReportedDataset"
    _write_timelapse_source_dataset(
        tmp_path,
        dataset_name,
        file_name="17h23m11s_rec_scan00_CAM.hdf5_multi.0.reconstruction.tiff",
        n_timepoints=31,
    )

    cfg = {
        "dataset_name": dataset_name,
        "pipeline": "recon_timelapse_simple",
        "data_type": "recon",
        "fmt": "timelapse",
        "pipeline_cfg": {},
        "log": {"enabled": True},
        "split": {"enabled": False},
    }

    stream = io.StringIO()
    reporter = ConsolePreprocessReporter(stream=stream)

    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute(reporter=reporter)

    assert result.n_files == 1
    output = stream.getvalue()
    assert "[1] 17h23m11s_rec_scan00_CAM.hdf5_multi.0.reconstruction.tiff -> c00_t31.tif (split=train)" in output

    manifest_path = tmp_path / dataset_name / "preprocess" / "recon" / settings.data_cfg.logs["recon_preprocess"]
    manifest = load_yaml(manifest_path)
    assert manifest["summary"]["split"]["train"] == {
        "count": 1,
        "source_names": ["17h23m11s_rec_scan00_CAM.hdf5_multi.0.reconstruction.tiff"],
        "output_names": ["c00_t31.tif"],
    }


def test_preprocess_run_refuses_to_overwrite_existing_content_without_approval(tmp_path: Path):
    dataset_name = "ExistingDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif"])
    preprocess_dir = tmp_path / dataset_name / "preprocess" / "recon"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    (preprocess_dir / "stale.txt").write_text("stale", encoding="utf-8")

    cfg = _single_cfg(dataset_name)
    run = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path))

    with pytest.raises(FileExistsError, match="Existing preprocess output detected"):
        run.execute()


def test_preprocess_run_allows_empty_existing_directory(tmp_path: Path):
    dataset_name = "EmptyExistingDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif"])
    preprocess_dir = tmp_path / dataset_name / "preprocess" / "recon"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    cfg = _single_cfg(dataset_name)
    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()

    assert result.n_files == 1
    assert (preprocess_dir / "c00.tif").exists()


def test_preprocess_failure_summary_tracks_only_processed_items(tmp_path: Path):
    dataset_name = "BrokenDataset"
    _write_broken_single_source_dataset(tmp_path, dataset_name)

    cfg = _single_cfg(dataset_name)
    cfg["split"] = {
        "enabled": True,
        "mode": "manual",
        "manual": {
            "match_by": "source_name",
            "val": ["img_b.tif"],
            "test": [],
        },
    }

    stream = io.StringIO()
    reporter = ConsolePreprocessReporter(stream=stream)
    run = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path))

    with pytest.raises(ValueError, match="expects 2D images"):
        run.execute(reporter=reporter)

    manifest_path = tmp_path / dataset_name / "preprocess" / "recon" / settings.data_cfg.logs["recon_preprocess"]
    manifest = load_yaml(manifest_path)
    assert manifest["status"] == "failed"
    assert manifest["summary"]["split"]["counts"] == {"train": 1, "val": 0, "test": 0}
    assert manifest["summary"]["split"]["train"] == {
        "count": 1,
        "source_names": ["img_a.tif"],
        "output_names": ["c00.tif"],
    }
    assert manifest["summary"]["split"]["val"] == {
        "count": 0,
        "source_names": [],
        "output_names": [],
    }
    assert manifest["error"]["type"] == "ValueError"

    output = stream.getvalue()
    assert "status: failed" in output
    assert "files written: 1" in output
    assert "validation images (0): none" in output
    assert "error: ValueError:" in output


def test_preprocess_run_can_reuse_split_from_existing_manifest(tmp_path: Path):
    source_dataset = "SourceDataset"
    target_dataset = "TargetDataset"
    file_names = ["img_a.tif", "img_b.tif", "img_c.tif"]
    _write_single_source_dataset(tmp_path, source_dataset, file_names)
    _write_single_source_dataset(tmp_path, target_dataset, file_names)

    source_cfg = _single_cfg(source_dataset)
    source_cfg["split"] = {
        "enabled": True,
        "mode": "manual",
        "manual": {
            "match_by": "source_name",
            "val": ["img_b.tif"],
            "test": ["img_c.tif"],
        },
    }
    PreprocessRun.from_cfg(source_cfg, paths=DummyPaths(tmp_path)).execute()

    target_cfg = _single_cfg(target_dataset)
    target_cfg["split"] = {
        "enabled": True,
        "mode": "reuse",
        "reuse": {
            "dataset_name": source_dataset,
            "data_type": "recon",
            "match_by": "source_name",
        },
    }

    result = PreprocessRun.from_cfg(target_cfg, paths=DummyPaths(tmp_path)).execute()

    assert result.n_files == 3
    assert (tmp_path / target_dataset / "preprocess" / "recon" / "train" / "c00.tif").exists()
    assert (tmp_path / target_dataset / "preprocess" / "recon" / "val" / "c01.tif").exists()
    assert (tmp_path / target_dataset / "preprocess" / "recon" / "test" / "c02.tif").exists()
