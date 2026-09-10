from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from lisai.config import load_yaml, settings
from lisai.preprocess import PreprocessRun
from lisai.preprocess.pipelines.paired_single_recon import PairedSingleReconConfig
from lisai.preprocess.pipelines.recon_mltpl_snr import ReconMltplSnrConfig, ReconMltplSnrPipeline
from lisai.preprocess.pipelines.recon_timelapse_simple import (
    ReconTimelapseSimpleConfig,
    ReconTimelapseSimplePipeline,
)
from lisai.preprocess.pipelines.single import SingleReconConfig


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


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full((8, 8), value, dtype=np.uint16))


def test_single_recon_uses_base_for_input_and_auxiliary(tmp_path: Path):
    dataset = "SingleBase"
    dump = tmp_path / dataset / "dump" / "recon" / "gag"
    _write_image(dump / "resolft" / "cell01.tif", 10)
    _write_image(dump / "conf" / "cell01.tif", 20)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "base_subfolder": "gag",
            "input_subfolder": "resolft",
            "auxiliary_subfolders": {"conf": {"matching": "required"}},
        },
        "split": {"enabled": False},
    }

    PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()

    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert np.all(tifffile.imread(preprocess / "resolft" / "c00.tif") == 10)
    assert np.all(tifffile.imread(preprocess / "conf" / "c00.tif") == 20)

    registry = load_yaml(tmp_path / "dataset_registry.yml")
    assert registry[dataset]["outputs"]["recon"] == [
        {"key": "main", "path": "resolft", "role": "inp", "axes": "YX"},
        {"key": "conf", "path": "conf", "role": "aux", "axes": "YX"},
    ]
    assert registry[dataset]["defaults"]["recon"]["input"] == "resolft"


def test_paired_single_recon_uses_common_base_for_all_roles(tmp_path: Path):
    dataset = "PairedBase"
    dump = tmp_path / dataset / "dump" / "recon" / "gag"
    _write_image(dump / "low" / "cell01.tif", 10)
    _write_image(dump / "high" / "cell01.tif", 20)
    _write_image(dump / "conf" / "cell01.tif", 30)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "paired_single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "base_subfolder": "gag",
            "input_subfolder": "low",
            "gt_subfolder": "high",
            "auxiliary_subfolders": {"conf": {"matching": "required"}},
        },
        "split": {"enabled": False},
    }

    PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()

    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert np.all(tifffile.imread(preprocess / "low" / "c00.tif") == 10)
    assert np.all(tifffile.imread(preprocess / "high" / "c00.tif") == 20)
    assert np.all(tifffile.imread(preprocess / "conf" / "c00.tif") == 30)

    registry = load_yaml(tmp_path / "dataset_registry.yml")
    assert registry[dataset]["defaults"]["recon"] == {
        "input": "low",
        "target": None,
        "eval_gt": "high",
    }


def test_single_recon_output_subfolder_can_flatten_or_override(tmp_path: Path):
    dataset = "SingleOutputOverride"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "cell01.tif", 10)

    flatten_cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "output_subfolder": None,
        },
        "split": {"enabled": False},
    }

    PreprocessRun.from_cfg(flatten_cfg, paths=DummyPaths(tmp_path)).execute()
    preprocess = tmp_path / dataset / "preprocess" / "recon"
    assert (preprocess / "c00.tif").exists()

    registry = load_yaml(tmp_path / "dataset_registry.yml")
    assert registry[dataset]["outputs"]["recon"] == [
        {"key": "main", "path": "", "role": "inp", "axes": "YX"}
    ]
    assert registry[dataset]["defaults"]["recon"]["input"] == ""

    override_dataset = "SingleOutputNamed"
    _write_image(tmp_path / override_dataset / "dump" / "recon" / "resolft" / "cell01.tif", 20)
    override_cfg = {
        **flatten_cfg,
        "dataset_name": override_dataset,
        "pipeline_cfg": {
            "input_subfolder": "resolft",
            "output_subfolder": "primary",
        },
    }
    PreprocessRun.from_cfg(override_cfg, paths=DummyPaths(tmp_path)).execute()
    assert (tmp_path / override_dataset / "preprocess" / "recon" / "primary" / "c00.tif").exists()


def test_single_recon_legacy_dump_subfolder_keeps_root_output(tmp_path: Path):
    dataset = "SingleLegacyOutput"
    dump = tmp_path / dataset / "dump" / "recon"
    _write_image(dump / "resolft" / "cell01.tif", 10)

    cfg = {
        "dataset_name": dataset,
        "usage": "evaluation",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {"dump_subfolder": "resolft"},
        "split": {"enabled": False},
    }
    PreprocessRun.from_cfg(cfg, paths=DummyPaths(tmp_path)).execute()
    assert (tmp_path / dataset / "preprocess" / "recon" / "c00.tif").exists()


def test_legacy_dump_subfolder_conflicts_only_with_its_new_equivalent():
    with pytest.raises(ValueError, match="input_subfolder.*dump_subfolder|dump_subfolder.*input_subfolder"):
        SingleReconConfig(input_subfolder="resolft", dump_subfolder="legacy")

    # single_recon legacy dump_subfolder is the input selector, so a new common base is still unambiguous.
    cfg = SingleReconConfig(base_subfolder="gag", dump_subfolder="resolft")
    assert cfg.base_subfolder == "gag"

    with pytest.raises(ValueError, match="base_subfolder.*dump_subfolder|dump_subfolder.*base_subfolder"):
        PairedSingleReconConfig(base_subfolder="gag", dump_subfolder="legacy")

    with pytest.raises(ValueError, match="base_subfolder.*dump_subfolder|dump_subfolder.*base_subfolder"):
        ReconTimelapseSimpleConfig(base_subfolder="gag", dump_subfolder="legacy")

    with pytest.raises(ValueError, match="base_subfolder.*dump_subfolder|dump_subfolder.*base_subfolder"):
        ReconMltplSnrConfig(base_subfolder="gag", dump_subfolder="legacy")


def test_timelapse_and_mltpl_snr_use_base_then_input_subfolder(tmp_path: Path):
    paths = DummyPaths(tmp_path)
    run = SimpleNamespace(dataset_name="Data", data_type="recon", paths=paths)
    expected = tmp_path / "Data" / "dump" / "recon" / "group" / "primary"

    timelapse = ReconTimelapseSimplePipeline(
        ReconTimelapseSimpleConfig(base_subfolder="group", input_subfolder="primary")
    )
    assert timelapse.build_source(run=run).root == expected

    mltpl = ReconMltplSnrPipeline(
        ReconMltplSnrConfig(base_subfolder="group", input_subfolder="primary")
    )
    assert mltpl.build_source(run=run).root == expected


def test_legacy_dump_subfolder_keeps_previous_pipeline_path_meaning(tmp_path: Path):
    paths = DummyPaths(tmp_path)
    run = SimpleNamespace(dataset_name="Data", data_type="recon", paths=paths)

    timelapse = ReconTimelapseSimplePipeline(ReconTimelapseSimpleConfig(dump_subfolder="legacy"))
    assert timelapse.build_source(run=run).root == tmp_path / "Data" / "dump" / "recon" / "legacy"

    mltpl = ReconMltplSnrPipeline(ReconMltplSnrConfig(dump_subfolder="legacy"))
    assert mltpl.build_source(run=run).root == tmp_path / "Data" / "dump" / "recon" / "legacy"
