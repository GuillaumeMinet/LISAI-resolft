"""APP-1: inference selections and TIFF I/O without Qt or ImProcess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lisai.application.inference import resolve_inference_selection
from lisai.application.inference_io import (
    InferenceProvenance,
    load_inference_tiff,
    save_prediction_tiff,
)
from lisai.evaluation.inference.results import PredictionInfo
from lisai.infra.paths import Paths
from lisai.model_sources import RunRef
from lisai.promoted_models.registry import (
    register_promoted_model,
    resolve_registered_model_dir,
)
from lisai.promoted_models.schema import PromotedModelRegistryEntry


def _register(settings, name="demo"):
    paths = Paths(settings)
    model_folder = paths.promoted_models_root() / name
    model_folder.mkdir(parents=True, exist_ok=True)
    register_promoted_model(
        name=name,
        entry=PromotedModelRegistryEntry(
            source_run_id="ABC123", path=name,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        paths=paths,
    )
    return model_folder


def test_installed_model_workspace_isolation(settings, tmp_path):
    from lisai.application.workspace import create_workspace

    folder = _register(settings)
    selection = resolve_inference_selection(settings=settings, source="installed", identifier="demo")
    assert selection.model == "demo"
    assert selection.source == "installed"
    assert selection.identifier == "demo"
    assert selection.overrides.model_dump(exclude_unset=True) == {}

    other = create_workspace(tmp_path / "other_workspace")
    with pytest.raises(ValueError, match="not installed"):
        resolve_inference_selection(settings=other, source="installed", identifier="demo")

    folder.rmdir()
    with pytest.raises(FileNotFoundError, match="directory is missing"):
        resolve_inference_selection(settings=settings, source="installed", identifier="demo")
    with pytest.raises(KeyError, match="Unknown"):
        resolve_registered_model_dir("missing", paths=Paths(settings))


def test_run_selection_and_checkpoint_availability(settings, monkeypatch):
    import lisai.application.catalogs as catalogs

    monkeypatch.setattr(catalogs, "list_usable_training_runs", lambda _settings: [
        {"run_id": "ABC123", "checkpoints": {"best": False, "last": True}},
    ])
    with pytest.raises(FileNotFoundError, match="best"):
        resolve_inference_selection(settings=settings, source="run", identifier="abc123")
    selected = resolve_inference_selection(
        settings=settings, source="run", identifier="abc123", checkpoint="last"
    )
    assert selected.model == RunRef("ABC123")
    assert selected.overrides.model_dump(exclude_unset=True) == {
        "checkpoint": {"best_or_last": "last"}
    }
    with pytest.raises(ValueError, match="Checkpoint"):
        resolve_inference_selection(settings=settings, source="run", identifier="ABC123",
                                    checkpoint="invalid")
    with pytest.raises(ValueError, match="not available"):
        resolve_inference_selection(settings=settings, source="run", identifier="UNKNOWN")


def test_sparse_normalization_tiling_crop_and_edges(settings):
    _register(settings)
    selected = resolve_inference_selection(
        settings=settings, source="installed", identifier="demo",
        tiling_size="model settings", crop_size=0,
        normalization_mode="model settings", temporal_edge_mode="",
        manual_gain=17.0, manual_offset=99.0, checkpoint="last",
    )
    assert selected.overrides.model_dump(exclude_unset=True) == {}
    selected = resolve_inference_selection(
        settings=settings, source="installed", identifier="demo",
        tiling_size="512", crop_size=128, normalization_mode="manual",
        manual_gain=9.94, manual_offset=-7.35,
        temporal_edge_mode="black_output",
    )
    assert selected.overrides.model_dump(exclude_unset=True) == {
        "input": {"normalization": {"mode": "manual", "gain": 9.94, "offset": -7.35}},
        "inference": {"crop_size": 128, "tiling_size": 512,
                      "temporal_edge_mode": "black_output"},
    }
    for tiling, expected in (("off", "off"), ("auto", "auto"), (256, 256)):
        opts = resolve_inference_selection(settings=settings, source="installed", identifier="demo",
                                           tiling_size=tiling).overrides
        assert opts.inference.tiling_size == expected
    for norm in ("training", "image"):
        opts = resolve_inference_selection(settings=settings, source="installed", identifier="demo",
                                           normalization_mode=norm).overrides
        assert opts.model_dump(exclude_unset=True) == {"input": {"normalization": {"mode": norm}}}


@pytest.mark.parametrize("field,value", [
    ("tiling_size", "invalid"), ("tiling_size", -2), ("tiling_size", True),
    ("crop_size", -1), ("crop_size", 1.0), ("crop_size", 1.5), ("crop_size", True),
    ("normalization_mode", "other"), ("manual_gain", -1),
    ("temporal_edge_mode", "skip"),
])
def test_invalid_overrides(settings, field, value):
    _register(settings)
    kwargs = {field: value}
    if field == "manual_gain":
        kwargs["normalization_mode"] = "manual"
    with pytest.raises(ValueError):
        resolve_inference_selection(settings=settings, source="installed", identifier="demo", **kwargs)


def test_selection_requires_real_settings(settings):
    with pytest.raises(TypeError, match="Settings"):
        resolve_inference_selection(settings=None, source="installed", identifier="demo")
    with pytest.raises(ValueError, match="Model source"):
        resolve_inference_selection(settings=settings, source="unknown", identifier="demo")
    with pytest.raises(ValueError, match="Select"):
        resolve_inference_selection(settings=settings, source="installed", identifier="")


@pytest.mark.parametrize("axes,shape", [
    ("YX", (23, 31)), ("TYX", (4, 23, 31)),
])
def test_load_explicit_tiff_axes(tmp_path, axes, shape):
    source = tmp_path / "input.tiff"
    pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    tifffile.imwrite(source, pixels, metadata={"axes": axes}, photometric="minisblack")
    loaded, meta = load_inference_tiff(source)
    np.testing.assert_array_equal(loaded, pixels)
    assert meta.axes == axes
    assert meta.path == source.resolve()
    assert meta.shape == shape
    assert meta.dtype == "uint16"


@pytest.mark.parametrize("axes", ["ZYX", "CYX", None])
def test_load_rejects_non_tyx_or_ambiguous_stacks(tmp_path, axes):
    source = tmp_path / "untrusted.tif"
    pix = np.zeros((6, 20, 22), dtype=np.uint16)
    tifffile.imwrite(source, pix, metadata={"axes": axes} if axes else None,
                     photometric="minisblack")
    with pytest.raises(ValueError, match="Unsupported TIFF axes"):
        load_inference_tiff(source)


def test_load_rejects_multi_series_and_missing_file(tmp_path):
    source = tmp_path / "multiple.tif"
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.zeros((20, 22), dtype=np.uint16), metadata={"axes": "YX"})
        writer.write(np.ones((20, 22), dtype=np.uint16), metadata={"axes": "YX"})
    with pytest.raises(ValueError, match="single TIFF"):
        load_inference_tiff(source)
    with pytest.raises(FileNotFoundError):
        load_inference_tiff(tmp_path / "missing.tif")


@pytest.mark.parametrize("axes,shape,info", [
    ("YX", (16, 18), PredictionInfo((0,), ())),
    ("TYX", (3, 16, 18), PredictionInfo((2, 3, 4), ())),
    ("TYX", (5, 16, 18), PredictionInfo((0, 1, 2, 3, 4), (0, 4))),
])
def test_save_prediction_with_embedded_provenance(tmp_path, axes, shape, info):
    output = tmp_path / "results" / "prediction.tif"
    pixels = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    original = pixels.copy()
    from lisai.config.models.inference import ApplyOverrides, ApplyInferenceOverrides

    prov = InferenceProvenance(
        source_path=tmp_path / "raw.tif", model=RunRef("Run-ID"),
        overrides=ApplyOverrides(inference=ApplyInferenceOverrides(tiling_size="off")),
    )
    saved = save_prediction_tiff(output, pixels, axes=axes, info=info, provenance=prov)
    assert saved == output
    np.testing.assert_array_equal(pixels, original)
    with tifffile.TiffFile(output) as tiff:
        assert tiff.series[0].axes == axes
        np.testing.assert_array_equal(tiff.asarray(), pixels.astype(np.float32))
        embedded = json.loads(tiff.pages[0].description)
        meta = embedded["lisai"]
        assert meta["source_frame_indices"] == list(info.source_frame_indices)
        assert meta["black_output_indices"] == list(info.black_output_indices)
        assert meta["model_source"] == "run"
        assert meta["model"] == "RUN-ID"
        assert meta["overrides"] == {"inference": {"tiling_size": "off"}}
        assert isinstance(meta["lisai_version"], str)

    with pytest.raises(FileExistsError):
        save_prediction_tiff(output, np.ones(shape), axes=axes, info=info, provenance=prov)
    assert np.array_equal(tifffile.imread(output), pixels.astype(np.float32))
    assert save_prediction_tiff(output, np.ones(shape), axes=axes, info=info,
                                provenance=prov, overwrite=True) == output
    np.testing.assert_array_equal(tifffile.imread(output), 1.0)


@pytest.mark.parametrize("info", [
    PredictionInfo((0, 1), ()), PredictionInfo((0,), (1,)),
    PredictionInfo((-1,), ()), PredictionInfo((0,), (0,)),
])
def test_save_rejects_bad_single_frame_alignment(tmp_path, info):
    with pytest.raises(ValueError):
        save_prediction_tiff(tmp_path / "out.tif", np.zeros((16, 16)),
                             axes="YX", info=info,
                             provenance=InferenceProvenance("source.tif", "model"))
    assert not (tmp_path / "out.tif").exists()


def test_save_does_not_touch_existing_file_on_validation_failure(tmp_path):
    output = tmp_path / "out.tif"
    output.write_bytes(b"original")
    with pytest.raises(ValueError):
        save_prediction_tiff(output, np.zeros((16, 16, 3)), axes="CYX",
                             info=PredictionInfo((0,)),
                             provenance=InferenceProvenance("raw.tif", "model"),
                             overwrite=True)
    assert output.read_bytes() == b"original"


def test_application_import_in_clean_interpreter_without_qt(tmp_path):
    script = ("import sys; from lisai.application import resolve_inference_selection, "
              "load_inference_tiff; assert 'qtpy' not in sys.modules; "
              "assert 'imswitch' not in sys.modules")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, env=env)
    assert result.returncode == 0, result.stderr
