"""R3: generic saved-YAML path is not tied to Quick HDN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from lisai.application import (
    create_workspace, plan_for_training_config, save_training_yaml_text,
    training_command,
)
from lisai.data.import_training import import_training_dataset


@pytest.fixture
def workspace(tmp_path):
    settings = create_workspace(tmp_path / "ws")
    import_training_dataset(
        [np.ones((3, 64, 64), dtype=np.uint16), np.ones((3, 64, 64), dtype=np.uint16) * 2],
        settings=settings, dataset_name="mito", data_format="timelapse",
    )
    return settings


@pytest.mark.parametrize("example", ["vim_upsamp_singleframe.yml", "vim_upsamp_multiframes_n5.yml"])
def test_general_saved_yaml_accepts_upsampling(workspace, example):
    source = Path(__file__).resolve().parents[2] / "src/lisai/resources/configs/training/examples" / example
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["experiment"]["exp_name"] = f"gui_{example.removesuffix('.yml')}"
    document["data"].update(
        dataset_name="mito", prep_before=False, already_split=False,
        paired=False, input="raw", target=None, data_format="timelapse",
    )
    document["routing"]["data_subfolder"] = ""
    saved = save_training_yaml_text(yaml.safe_dump(document), example, settings=workspace)
    plan = plan_for_training_config(saved, settings=workspace)
    assert plan.mode == "train"
    assert plan.dataset_name == "mito"
    assert plan.models_subfolder == "Upsamp"
    assert plan.run_dir is not None
    cmd = training_command(workspace, plan, interpreter="python-test")
    assert cmd[0] == "python-test"
    assert cmd[cmd.index("--config") + 1] == str(saved)

    plan.run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        plan_for_training_config(saved, settings=workspace)


def test_gui_preflight_rejects_destructive_overwrite(workspace):
    source = Path(__file__).resolve().parents[2] / "src/lisai/resources/configs/training/examples/vim_upsamp_singleframe.yml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["experiment"].update(exp_name="gui_overwrite", overwrite=True)
    document["data"].update(dataset_name="mito", prep_before=False, already_split=False,
                            paired=False, input="raw", target=None, data_format="timelapse")
    document["routing"]["data_subfolder"] = ""
    saved = save_training_yaml_text(yaml.safe_dump(document), "overwrite.yml", settings=workspace)
    with pytest.raises(ValueError, match="overwrite=true"):
        plan_for_training_config(saved, settings=workspace)
