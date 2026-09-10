from __future__ import annotations

from pathlib import Path
import io

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


def _write_image(path: Path, value: int, shape: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full(shape, value, dtype=np.uint16))


def test_single_recon_saves_required_auxiliary_and_registers_role(tmp_path: Path):
    dataset = "SingleAux"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "cell01.tif", 10)
    _write_image(dump / "confocal" / "cell01.tif", 20)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "dump_subfolder": "resolft",
            "auxiliary_subfolders": {
                "confocal": {"matching": "required"},
            },
        },
        "split": {"enabled": False},
    }

    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()
    assert result.n_files == 1

    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert np.all(tifffile.imread(preprocess / "c00.tif") == 10)
    assert np.all(tifffile.imread(preprocess / "confocal" / "c00.tif") == 20)

    registry = load_yaml(tmp_path / "dataset_registry.yml")
    assert registry[dataset]["outputs"]["recon"] == [
        {"key": "main", "path": "", "role": "inp", "axes": "YX"},
        {"key": "confocal", "path": "confocal", "role": "aux", "axes": "YX"},
    ]
    assert registry[dataset]["defaults"]["recon"] == {
        "input": "",
        "target": None,
        "eval_gt": None,
    }

    manifest = load_yaml(preprocess / settings.data_cfg.logs["recon_preprocess"])
    assert manifest["items"][0]["auxiliary_source_paths"] == {
        "confocal": str(dump / "confocal" / "cell01.tif")
    }


def test_single_recon_required_auxiliary_must_match_primary_filename(tmp_path: Path):
    dataset = "SingleAuxMissing"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "cell01.tif", 10)
    _write_image(dump / "confocal" / "other.tif", 20)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "dump_subfolder": "resolft",
            "auxiliary_subfolders": {"confocal": {"matching": "required"}},
        },
        "split": {"enabled": False},
    }

    with pytest.raises(ValueError, match="Missing required auxiliary 'confocal'.*cell01.tif"):
        PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()


def test_paired_single_recon_saves_auxiliary_without_registering_it_with_primary_pair(tmp_path: Path):
    dataset = "PairedAux"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "low" / "cell01.tif", 10, shape=(10, 10))
    _write_image(dump / "high" / "cell01.tif", 20, shape=(10, 10))
    _write_image(dump / "confocal" / "cell01.tif", 30, shape=(10, 10))

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "paired_single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "low",
            "gt_subfolder": "high",
            "registration": False,
            "crop_size": 8,
            "auxiliary_subfolders": {
                "confocal": {"matching": "required"},
            },
        },
        "split": {"enabled": False},
    }

    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()
    assert result.n_files == 1

    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert tifffile.imread(preprocess / "low" / "c00.tif").shape == (8, 8)
    assert tifffile.imread(preprocess / "high" / "c00.tif").shape == (8, 8)
    assert tifffile.imread(preprocess / "confocal" / "c00.tif").shape == (8, 8)

    registry = load_yaml(tmp_path / "dataset_registry.yml")
    assert registry[dataset]["outputs"]["recon"] == [
        {"key": "inp", "path": "low", "role": "inp", "axes": "YX"},
        {"key": "gt", "path": "high", "role": "gt", "axes": "YX"},
        {"key": "confocal", "path": "confocal", "role": "aux", "axes": "YX"},
    ]
    assert registry[dataset]["defaults"]["recon"] == {
        "input": "low",
        "target": None,
        "eval_gt": "high",
    }


def test_auxiliary_matching_must_be_explicit(tmp_path: Path):
    dataset = "MissingMatching"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "cell01.tif", 10)
    _write_image(dump / "confocal" / "cell01.tif", 20)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {"confocal": {}},
        },
        "split": {"enabled": False},
    }

    with pytest.raises(ValueError, match="requires an explicit 'matching' value"):
        PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()


def test_optional_filename_matching_keeps_unmatched_primary_and_reports_count(tmp_path: Path):
    dataset = "OptionalFilenameAux"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "cell01.tif", 10)
    _write_image(dump / "resolft" / "cell02.tif", 11)
    _write_image(dump / "confocal" / "cell01.tif", 20)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {
                "confocal": {"matching": "optional"},
            },
        },
        "split": {"enabled": False},
    }

    stream = io.StringIO()
    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute(
        reporter=ConsolePreprocessReporter(stream=stream)
    )
    assert result.n_files == 2

    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert (preprocess / "resolft" / "c00.tif").exists()
    assert (preprocess / "resolft" / "c01.tif").exists()
    assert (preprocess / "confocal" / "c00.tif").exists()
    assert not (preprocess / "confocal" / "c01.tif").exists()
    assert "Auxiliary 'confocal': matched 1/2" in stream.getvalue()


def test_timestamp_matching_requires_max_time_delta(tmp_path: Path):
    dataset = "TimestampNeedsDelta"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "00h00m10s_primary.tif", 10)
    _write_image(dump / "confocal" / "00h00m09s_aux.tif", 20)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {
                "confocal": {
                    "matching": "required",
                    "match_by": "timestamp",
                }
            },
        },
        "split": {"enabled": False},
    }

    with pytest.raises(ValueError, match="requires max_time_delta_s"):
        PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()


def test_timestamp_matching_is_one_to_one(tmp_path: Path):
    dataset = "TimestampOneToOne"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "00h00m10s_primary.tif", 10)
    _write_image(dump / "resolft" / "00h00m11s_primary.tif", 11)
    _write_image(dump / "confocal" / "00h00m10s_aux.tif", 100)
    _write_image(dump / "confocal" / "00h00m20s_aux.tif", 200)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {
                "confocal": {
                    "matching": "required",
                    "match_by": "timestamp",
                    "max_time_delta_s": 15,
                }
            },
        },
        "split": {"enabled": False},
    }

    PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()
    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert np.all(tifffile.imread(preprocess / "confocal" / "c00.tif") == 100)
    assert np.all(tifffile.imread(preprocess / "confocal" / "c01.tif") == 200)


def test_timestamp_relation_after_primary_filters_candidates(tmp_path: Path):
    dataset = "TimestampRelation"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "00h00m10s_primary.tif", 10)
    _write_image(dump / "confocal" / "00h00m09s_aux.tif", 90)
    _write_image(dump / "confocal" / "00h00m11s_aux.tif", 110)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {
                "confocal": {
                    "matching": "required",
                    "match_by": "timestamp",
                    "max_time_delta_s": 5,
                    "timestamp_relation": "after_primary",
                }
            },
        },
        "split": {"enabled": False},
    }

    PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()
    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert np.all(tifffile.imread(preprocess / "confocal" / "c00.tif") == 110)


def test_optional_timestamp_matching_keeps_primary_without_close_auxiliary(tmp_path: Path):
    dataset = "OptionalTimestampAux"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "00h00m10s_primary.tif", 10)
    _write_image(dump / "resolft" / "00h00m30s_primary.tif", 30)
    _write_image(dump / "confocal" / "00h00m11s_aux.tif", 110)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {
                "confocal": {
                    "matching": "optional",
                    "match_by": "timestamp",
                    "max_time_delta_s": 3,
                }
            },
        },
        "split": {"enabled": False},
    }

    result = PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()
    assert result.n_files == 2

    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert (preprocess / "resolft" / "c00.tif").exists()
    assert (preprocess / "resolft" / "c01.tif").exists()
    assert (preprocess / "confocal" / "c00.tif").exists()
    assert not (preprocess / "confocal" / "c01.tif").exists()
