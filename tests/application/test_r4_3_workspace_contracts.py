"""R4-C non-Qt contracts: resource discovery, run matching, job values."""
from __future__ import annotations

from pathlib import Path

from lisai.application import JobSnapshot, TrainingLaunchRequest, list_noise_model_entries
from lisai.application import catalogs
from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.infra.paths import Paths


def test_noise_browser_does_not_claim_calibration_is_valid(tmp_path):
    root = tmp_path / "lisai"
    initialize_workspace(root)
    settings = Settings.from_workspace(root)
    paths = Paths(settings)
    model = paths.noise_model_path(noiseModel_name="camera")
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"not a valid scientific calibration")
    entries = list_noise_model_entries(settings)
    camera = next(entry for entry in entries if entry.name == "camera")
    assert camera.model_present
    assert not camera.normalization_present
    assert not camera.usable
    norm = paths.noise_model_norm_prm_path(noiseModel_name="camera")
    norm.write_text("{}")
    camera = next(entry for entry in list_noise_model_entries(settings) if entry.name == "camera")
    assert camera.usable  # file presence only, never scientific validation


def test_completion_lookup_matches_numbered_folders_and_avoids_ambiguity(monkeypatch, tmp_path):
    runs = [
        {"run_id": "old", "dataset": "vim", "name": "exp", "run_dir": tmp_path / "exp_00"},
        {"run_id": "fresh", "dataset": "vim", "name": "exp_01", "run_dir": tmp_path / "exp_01"},
        {"run_id": "other", "dataset": "mito", "name": "exp_02", "run_dir": tmp_path / "exp_02"},
    ]
    monkeypatch.setattr(catalogs, "list_usable_training_runs", lambda _settings: runs)
    query = dict(dataset_name="vim", experiment_name="exp", old_run_ids={"old"})
    assert catalogs.find_new_completed_run(object(), **query) == "fresh"
    runs.append({"run_id": "another", "dataset": "vim", "name": "exp_03",
                 "run_dir": tmp_path / "exp_03"})
    assert catalogs.find_new_completed_run(object(), **query) is None


def test_training_job_contract_is_independent_of_qt(tmp_path):
    request = TrainingLaunchRequest(
        tmp_path / "experiment.yml", tmp_path / "workspace", tmp_path / "configs", "train")
    state = JobSnapshot(None, "running", None, None, None, "Running")
    assert request.mode == "train"
    assert state.progress is None
    assert state.run_id is None
