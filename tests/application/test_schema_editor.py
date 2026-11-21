"""First-pass scientific field guidance remains backed by real LISAI schema."""
from __future__ import annotations

import pytest

from lisai.application.schema_editor import training_editor_fields, update_yaml_field
from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "lisai"
    initialize_workspace(root)
    return Settings.from_workspace(root)


@pytest.mark.parametrize("task,essential", [
    ("denoising_hdn", {"betaKL", "supervised"}),
    ("upsamp_single_frame", {"sampling_ratio", "upsampling_factor"}),
    ("upsamp_multiframes", {"temporal_window", "upsampling_factor"}),
    ("denoising_care", {"loss"}),
])
def test_scientific_fields_prioritized_from_real_schema(settings, task, essential):
    sections = training_editor_fields(settings, task)
    names = {field.path[-1]: field for field in sections["Experiment"]}
    assert essential <= names.keys()
    for name in essential:
        assert names[name].description
        assert names[name].path[:2] == ("experiment", "task")
    assert sections["Optimization"][0].path == ("training", "n_epochs")
    if task == "upsamp_single_frame":
        assert set(names["sampling_ratio"].choices) == {0.25, 0.5, 0.75}


def test_round_trip_guided_edits_leave_raw_yaml_comments_and_unknown_fields():
    pytest.importorskip("ruamel.yaml")
    text = """# experiment header
experiment:
  task:
    name: denoising_hdn
    betaKL: 0.5  # important scientific choice
    supervised: false
  exp_name: first
training:
  n_epochs: 10
custom_extension: [a, b]  # unseen by guided UI
"""
    changed = update_yaml_field(text, ("experiment", "task", "betaKL"), 0.3)
    assert "# experiment header" in changed
    assert "# important scientific choice" in changed
    assert "custom_extension: [a, b]" in changed
    assert "0.3" in changed
    assert "0.5" not in changed
    assert "supervised: false" in changed


def test_round_trip_rejects_invalid_shape():
    pytest.importorskip("ruamel.yaml")
    with pytest.raises(ValueError, match="must be a mapping"):
        update_yaml_field("- list\n", ("training", "n_epochs"), 4)
    with pytest.raises(ValueError, match="not an object"):
        update_yaml_field("training: 42\n", ("training", "n_epochs"), 4)
