"""R4-B: reusable data/catalog workflows with no Qt or implicit writes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from lisai.application import (
    TrainingConfigDocument, TrainingRecording, apply_dataset_change,
    create_document_from_source, create_workspace, list_config_sources,
    list_training_datasets, preview_dataset_change, training_recording_sources,
)
from lisai.config.catalog import (
    build_training_config_from_preset, create_training_config_from_preset,
    build_training_config_from_template, create_training_config_from_template,
)
from lisai.data.dataset_registry import load_dataset_registry, save_dataset_registry
from lisai.data.import_training import import_training_dataset
from lisai.infra.paths import Paths


@pytest.fixture
def ws(tmp_path):
    return create_workspace(tmp_path / "ws")


def test_general_dataset_catalog_does_not_filter_out_preprocessed_datasets(ws):
    import_training_dataset(
        [np.ones((2, 32, 32), dtype=np.uint16), np.ones((2, 32, 32), dtype=np.uint16)],
        settings=ws, dataset_name="raw_timelapses", data_format="timelapse",
    )
    path = Paths(ws).dataset_registry_path()
    registry = load_dataset_registry(path)
    registry["prepared"] = {
        "usage": "training", "for_training": True, "data_format": "single",
        "description": "Preprocessed registered dataset",
        "defaults": {"recon": {"input": "resolft", "target": None}},
        "outputs": {"recon": [{"key": "resolft", "path": "resolft", "role": "inp", "axes": "YX"}]},
    }
    registry["excluded"] = {"usage": "inference", "data_format": "single"}
    registry["non_training"] = {"usage": "training", "for_training": False,
                                "data_format": "single"}
    save_dataset_registry(registry, path)
    entries = {entry.name: entry for entry in list_training_datasets(ws)}
    assert {"raw_timelapses", "prepared", "non_training"} <= entries.keys()
    assert "excluded" not in entries
    assert entries["raw_timelapses"].quick_hdn_eligible
    assert not entries["prepared"].quick_hdn_eligible
    assert entries["prepared"].data_types == ("recon",)
    assert entries["prepared"].available_roles == ("inp",)
    assert not entries["non_training"].for_training


def test_dataset_candidate_is_explicit_preview_and_apply(ws):
    import_training_dataset(
        [np.ones((3, 32, 32), dtype=np.uint16), np.ones((3, 32, 32), dtype=np.uint16)],
        settings=ws, dataset_name="mito", data_format="timelapse",
    )
    document, _ = create_document_from_source(
        ws, "presets/upsamp_multiframes", dataset_selection="old", experiment_name="test"
    )
    before = document.yaml_text
    proposed = preview_dataset_change(document, ws, "mito")
    assert proposed.dataset_name == "mito"
    assert document.yaml_text == before  # no mutation on browse/preview
    changed_fields = {".".join(path): new for path, _old, new in proposed.changes}
    assert changed_fields["data.dataset_name"] == "mito"
    assert changed_fields["data.input"] == "raw"
    assert document.read_field(("data", "data_format")) == "timelapse"
    assert "data.data_format" not in changed_fields  # unchanged values are not rewritten
    assert changed_fields["routing.data_subfolder"] == ""
    assert changed_fields["data.prep_before"] is False
    assert changed_fields["data.already_split"] is False
    apply_dataset_change(document, proposed)
    assert document.read_field(("data", "dataset_name")) == "mito"
    assert document.read_field(("data", "input")) == "raw"
    assert "#" in document.yaml_text  # preserve generated section comments
    assert document.dirty
    with pytest.raises(RuntimeError, match="changed after dataset preview"):
        apply_dataset_change(document, proposed)


def test_dataset_change_does_not_guess_ambiguous_role_or_type(ws):
    path = Paths(ws).dataset_registry_path()
    save_dataset_registry({"ambiguous": {
        "usage": "training", "data_format": "single",
        "outputs": {
            "resolft": [{"path": "a", "role": "inp"}, {"path": "b", "role": "inp"}],
            "conf": [{"path": "c", "role": "inp"}],
        },
    }}, path)
    doc = TrainingConfigDocument.from_text("data:\n  dataset_name: original\n  input: previous\n")
    with pytest.raises(ValueError, match="Select a data type"):
        preview_dataset_change(doc, ws, "ambiguous")
    proposed = preview_dataset_change(doc, ws, "ambiguous", data_type="resolft")
    assert any("input" in warning for warning in proposed.warnings)
    assert all(path != ("data", "input") for path, _old, _new in proposed.changes)
    with pytest.raises(ValueError, match="not registered for training"):
        preview_dataset_change(doc, ws, "missing")


def test_recording_inputs_are_host_neutral_and_preserve_timelapses(ws, tmp_path):
    data = np.ones((4, 16, 16), dtype=np.uint16)
    recordings = [TrainingRecording("frame_1", data, ("T", "Y", "X"))]
    sources = training_recording_sources(recordings, [], "timelapse")
    assert len(sources) == 1 and sources[0][1].shape == (4, 16, 16)
    with pytest.raises(ValueError, match="YX"):
        training_recording_sources(recordings, [], "single")
    with pytest.raises(ValueError, match="At least 2"):
        training_recording_sources(recordings, [], "timelapse", minimum_recordings=2)
    with pytest.raises(ValueError, match="share a TIFF filename"):
        training_recording_sources(recordings * 2, [], "timelapse")


@pytest.mark.parametrize("source", [
    "presets/denoising_hdn_unsup", "presets/denoising_hdn_sup",
    "presets/denoising_care", "presets/denoising_unetrcan",
    "presets/upsamp_single_frame", "presets/upsamp_multiframes",
    "templates/base_training",
])
def test_packaged_config_creation_uses_shared_builder_without_writing(ws, source):
    doc, warnings = create_document_from_source(
        ws, source, dataset_selection="mito", experiment_name="common"
    )
    assert doc.save_path is None and doc.dirty
    assert doc.source_ref is not None
    assert doc.read_field(("data", "dataset_name")) == "mito"
    assert doc.read_field(("experiment", "exp_name")) == "common"
    if source.startswith("presets/"):
        expected, other = build_training_config_from_preset(
            source, dataset_name="mito", exp_name="common", stg=ws)
    else:
        expected, other = build_training_config_from_template(
            source, dataset_name="mito", exp_name="common", stg=ws)
    assert yaml.safe_load(doc.yaml_text) == expected
    assert warnings == other
    assert doc.validate_for_save(ws)  # no invented input defaults for unknown registry
    assert not (ws.TRAINING_CONFIG_DIR / "common.yml").exists()
    destination = doc.save_as(ws, "science/common.yml")
    assert destination.is_file()
    assert yaml.safe_load(destination.read_text()) == expected
    expected_schema = ("experiment-template.schema.json" if source.startswith("templates/")
                       else "experiment.schema.json")
    assert destination.read_text().startswith(
        f"# yaml-language-server: $schema=../../schema/{expected_schema}")
    with pytest.raises(FileExistsError):
        doc.save_as(ws, "science/common.yml")


def test_existing_cli_creation_preserved_and_matches_draft(ws):
    doc, _ = create_document_from_source(
        ws, "presets/denoising_hdn_unsup", dataset_selection="mito", experiment_name="baseline"
    )
    path, _ = create_training_config_from_preset(
        "denoising_hdn_unsup", dataset_name="mito", exp_name="baseline", stg=ws)
    assert yaml.safe_load(path.read_text()) == yaml.safe_load(doc.yaml_text)
    with pytest.raises(FileExistsError):
        create_training_config_from_preset(
            "denoising_hdn_unsup", dataset_name="mito", exp_name="baseline", stg=ws)
    templ, _ = create_training_config_from_template(
        "base_training", dataset_name="mito", exp_name="templ", stg=ws)
    assert templ.is_file()


def test_examples_are_verbatim_readonly_sources_and_existing_files_are_separate(ws):
    entries = list_config_sources(ws)
    assert "examples/sweep_hdn_betaKL.yaml" not in {e.relative_path for e in entries}
    kinds = {e.kind for e in entries}
    assert {"preset", "template", "example"} <= kinds
    doc, warnings = create_document_from_source(
        ws, "examples/vim_upsamp_multiframes_n5", dataset_selection="different",
        experiment_name="override_not_allowed",
    )
    assert doc.source_kind == "example" and doc.save_path is None
    assert doc.read_field(("data", "dataset_name")) == "vim_live"
    assert doc.read_field(("experiment", "exp_name")) != "override_not_allowed"
    assert any("not applied" in warning for warning in warnings)
    assert "#" in doc.yaml_text
    with pytest.raises(ValueError, match="Save As"):
        doc.save(ws)
    saved = doc.save_as(ws, "example_copy.yml")
    assert saved.is_file()
    assert saved.read_text().startswith(
        "# yaml-language-server: $schema=../schema/experiment.schema.json")
    assert any(e.kind == "local" and e.relative_path == "example_copy.yml"
               for e in list_config_sources(ws))
    with pytest.raises(ValueError, match="existing writable"):
        create_document_from_source(ws, "example_copy.yml")
    with pytest.raises(FileNotFoundError):
        create_document_from_source(ws, "presets/not_a_config")
