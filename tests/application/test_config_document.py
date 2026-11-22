"""R4-A: one editable config document and separated authoring/execution rules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lisai.application import (
    TrainingConfigDocument, create_workspace, plan_for_training_config,
    quick_hdn_training_document, training_command,
)
from lisai.data.import_training import import_training_dataset
from lisai.infra.fs.folders import create_run_dir
from lisai.infra.paths import Paths
from lisai.training.quick_hdn import prepare_quick_hdn_training


@pytest.fixture
def ws(tmp_path):
    return create_workspace(tmp_path / "workspace")


def test_document_round_trip_edit_and_safe_save(ws):
    text = """# retain this header
experiment:
  exp_name: old
  task:
    name: denoising_hdn
    betaKL: 0.5  # scientific note
extra_custom_field: 42
"""
    # The authoring validator intentionally does not replace the scientific
    # schema; unknown top-level fields are *not* silently allowed on Save.
    with pytest.raises(ValueError, match="Unknown top-level"):
        TrainingConfigDocument.from_text(text).validate_for_save(ws)
    text = text.replace("extra_custom_field: 42", "training:\n  n_epochs: 10")
    doc = TrainingConfigDocument.from_text(text)
    assert doc.dirty
    assert doc.read_field(("experiment", "task", "betaKL")) == 0.5
    doc.update_field(("experiment", "task", "betaKL"), 0.3)
    assert "# retain this header" in doc.yaml_text
    assert "# scientific note" in doc.yaml_text
    assert doc.read_field(("experiment", "task", "betaKL")) == 0.3
    with pytest.raises(ValueError, match="Save As"):
        doc.save(ws)
    saved = doc.save_as(ws, "nested/experiment.yml")
    assert saved.is_file() and not doc.dirty
    assert doc.source_kind == "existing"
    assert doc.saved_text == doc.yaml_text
    assert "# scientific note" in saved.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        doc.save_as(ws, "nested/experiment.yml")
    fresh = TrainingConfigDocument.open(ws, saved)
    assert fresh.save_path == saved and not fresh.dirty
    fresh.update_field(("experiment", "task", "betaKL"), 0.2)
    assert fresh.dirty
    assert fresh.save(ws) == saved
    assert not fresh.dirty
    assert "betaKL: 0.2" in saved.read_text(encoding="utf-8")


def test_invalid_raw_buffer_never_overwrites_saved_file(ws):
    doc = TrainingConfigDocument.from_text("experiment:\n  exp_name: CHANGEME\n")
    path = doc.save_as(ws, "scratch.yml")
    original = path.read_bytes()
    doc.replace_text("experiment: [invalid")
    with pytest.raises(ValueError, match="syntax"):
        doc.save(ws)
    assert path.read_bytes() == original
    assert doc.yaml_text == "experiment: [invalid"
    with pytest.raises(ValueError, match="syntax"):
        doc.update_field(("experiment", "exp_name"), "new")
    assert doc.yaml_text == "experiment: [invalid"
    doc.replace_text("experiment:\n  exp_name: corrected\n")
    doc.save(ws)
    assert "corrected" in path.read_text()


def test_incomplete_template_can_be_saved_but_not_trained(ws):
    document = TrainingConfigDocument.from_source(ws, "templates/base_training")
    assert document.source_kind == "template"
    assert document.save_path is None
    assert "CHANGEME" in document.yaml_text or "<" in document.yaml_text
    warnings = document.validate_for_save(ws)
    assert warnings
    saved = document.save_as(ws, "pending.yml")
    assert saved.is_file() and document.save_path == saved
    with pytest.raises(ValueError, match="placeholders"):
        document.validate_for_training(ws)
    # Future edits can fill in placeholders without needing another Save As.
    document.update_field(("experiment", "exp_name"), "still_incomplete")
    document.save(ws)
    assert document.save_path == saved


def test_on_disk_conflict_requires_reload_or_save_as(ws):
    first = TrainingConfigDocument.from_text("experiment:\n  exp_name: first\n")
    path = first.save_as(ws, "one.yml")
    a = TrainingConfigDocument.open(ws, path)
    b = TrainingConfigDocument.open(ws, path)
    a.update_field(("experiment", "exp_name"), "a")
    a.save(ws)
    b.update_field(("experiment", "exp_name"), "b")
    with pytest.raises(RuntimeError, match="outside this editor"):
        b.save(ws)
    assert "exp_name: a" in path.read_text()
    assert "exp_name: b" in b.yaml_text
    separate = b.save_as(ws, "two.yml")
    assert "exp_name: b" in separate.read_text()
    path.unlink()
    a.update_field(("experiment", "exp_name"), "another")
    with pytest.raises(FileNotFoundError, match="Save As"):
        a.save(ws)


def test_builtin_source_is_read_only_and_paths_do_not_escape(ws, tmp_path):
    document = TrainingConfigDocument.from_source(ws, "presets/denoising_hdn_unsup")
    assert document.save_path is None and document.source_kind == "preset"
    with pytest.raises(ValueError, match="Save As"):
        document.save(ws)
    with pytest.raises(ValueError, match="inside"):
        document.save_as(ws, tmp_path / "outside.yml")
    assert document.save_as(ws, "derived.yml").is_file()
    with pytest.raises(ValueError, match="reserved"):
        document.save_as(ws, "presets/custom.yml")
    with pytest.raises(ValueError):
        TrainingConfigDocument.open(ws, "../../outside.yml")


def test_quick_hdn_draft_is_non_persisting_and_same_experiment_reusable(ws):
    paths = Paths(ws)
    noise_dir = paths.noise_model_dir(noiseModel_name="calibrated")
    noise_dir.mkdir(parents=True)
    np.savez(noise_dir / "GMM.npz", min_signal=np.array([0.0]), max_signal=np.array([255.0]),
             trained_weight=np.zeros((3, 2)), min_sigma=np.array([1.0]))
    (noise_dir / "norm_prm.json").write_text(
        json.dumps({"clip": 0, "normalize_data": False}), encoding="utf-8")
    import_training_dataset([np.ones((3, 64, 64), dtype=np.uint16),
                             np.ones((3, 64, 64), dtype=np.uint16) * 2],
                            settings=ws, dataset_name="mito", data_format="timelapse")
    opts = dict(settings=ws, dataset_name="mito", noise_model_name="calibrated",
                experiment_name="repeat_hdn", epochs=2, patch_size=32)
    doc, plan = quick_hdn_training_document(**opts)
    assert doc.dirty and doc.save_path is None
    assert not plan.config_path.exists()
    saved = doc.save_as(ws, "improcess/repeat_hdn.yml")
    assert plan_for_training_config(saved, settings=ws).run_dir is None
    first, _ = create_run_dir(paths, "mito", "repeat_hdn", "HDN")
    second, _ = create_run_dir(paths, "mito", "repeat_hdn", "HDN")
    assert first != second
    # Draft generation can be repeated against an existing config and run.
    again, _ = quick_hdn_training_document(**opts)
    assert again.save_path is None and again.dirty
    assert prepare_quick_hdn_training(**opts, write=False).config_path == plan.config_path
    with pytest.raises(FileExistsError):
        prepare_quick_hdn_training(**opts, write=True)
    assert plan_for_training_config(saved, settings=ws).run_dir is None


def test_preflight_keeps_workspace_identity(ws, tmp_path):
    from lisai.application import save_training_yaml_text

    example = Path(__file__).resolve().parents[2] / "src/lisai/resources/configs/training/examples/vim_upsamp_singleframe.yml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    data["data"].update(dataset_name="test", prep_before=False, already_split=False,
                        input="raw", paired=False, target=None, data_format="timelapse")
    saved = save_training_yaml_text(yaml.safe_dump(data), "sample.yml", settings=ws)
    plan = plan_for_training_config(saved, settings=ws)
    assert plan.run_dir is None
    assert plan.workspace_root == ws.WORKSPACE_ROOT
    assert training_command(ws, plan)[-1] == str(ws.CONFIGS_ROOT)
    second_ws = create_workspace(tmp_path / "another")
    with pytest.raises(ValueError, match="another workspace"):
        training_command(second_ws, plan)


def test_failed_guided_update_does_not_rewrite_malformed_objects():
    document = TrainingConfigDocument.from_text("experiment: 12\n")
    with pytest.raises(ValueError, match="not an object"):
        document.update_field(("experiment", "task", "betaKL"), 0.3)
    assert document.yaml_text == "experiment: 12\n"
