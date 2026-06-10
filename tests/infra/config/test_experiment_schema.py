from __future__ import annotations

import json
from pathlib import Path

import pytest

from lisai.config import load_yaml
from lisai.config.json_schema import (
    continue_training_json_schema,
    experiment_json_schema,
    experiment_template_json_schema,
    retrain_json_schema,
    write_continue_training_json_schema,
    write_experiment_json_schema,
    write_experiment_template_json_schema,
    write_retrain_json_schema,
)
from lisai.config.models import ContinueTrainingConfig, ExperimentConfig, RetrainConfig


@pytest.mark.parametrize(
    ("config_path", "model_cls"),
    [
        (Path("configs/training/examples/vim_hdn_unsup_betaKL05.yml"), ExperimentConfig),
        (Path("configs/training/examples/continue_training.yml"), ContinueTrainingConfig),
        (Path("configs/training/examples/retrain.yml"), RetrainConfig),
    ],
)
def test_training_yaml_examples_validate_against_mode_specific_authoring_schemas(config_path: Path, model_cls):
    cfg = load_yaml(config_path)

    validated = model_cls.model_validate(cfg)

    assert validated.experiment.mode



def test_current_upsamp_yaml_validates_timelapse_context_channels():
    cfg = load_yaml(Path("configs/training/examples/vim_upsamp_multiframes_n5.yml"))

    validated = ExperimentConfig.model_validate(cfg)

    assert validated.data.timelapse_prm is not None
    context_length = validated.data.timelapse_prm.context_length
    assert context_length == 5
    assert validated.model.architecture == "unet_rcan"
    assert validated.model.parameters.UNet_prm.in_channels == context_length
    assert validated.model.parameters.UNet_prm.out_channels == context_length



def test_experiment_json_schema_describes_train_authoring_shape_only():
    schema = experiment_json_schema()

    data_ref = schema["properties"]["data"]["$ref"].split("/")[-1]
    data_properties = schema["$defs"][data_ref]["properties"]
    assert "metadata" in schema["properties"]
    assert "data_dir" not in data_properties
    assert "dataset_info" not in data_properties
    assert "volumetric" not in data_properties
    assert "masking" not in data_properties
    assert "load_model" not in schema["properties"]



def test_continue_training_json_schema_only_exposes_resume_specific_roots():
    schema = continue_training_json_schema()
    properties = schema["properties"]

    assert "experiment" in properties
    assert "training" in properties
    assert "saving" in properties
    assert "tensorboard" in properties
    assert "load_model" in properties
    assert "data" not in properties
    assert "model" not in properties
    assert "routing" not in properties
    assert "loss_function" not in properties



def test_retrain_json_schema_exposes_transfer_learning_roots_but_not_model():
    schema = retrain_json_schema()
    properties = schema["properties"]

    assert "experiment" in properties
    assert "routing" in properties
    assert "data" in properties
    assert "normalization" in properties
    assert "noise_model" in properties
    assert "loss_function" in properties
    assert "load_model" in properties
    assert "model" not in properties



def test_write_training_json_schemas_write_json_files(tmp_path: Path):
    outputs = [
        (write_experiment_json_schema, tmp_path / "experiment.schema.json", "ExperimentConfig"),
        (
            write_experiment_template_json_schema,
            tmp_path / "experiment-template.schema.json",
            "ExperimentTemplateConfig",
        ),
        (write_continue_training_json_schema, tmp_path / "continue_training.schema.json", "ContinueTrainingConfig"),
        (write_retrain_json_schema, tmp_path / "retrain.schema.json", "RetrainConfig"),
    ]

    for writer, output_path, expected_title in outputs:
        written_path = writer(output_path)
        assert written_path == output_path
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["title"] == expected_title


def test_training_schema_exposes_new_controls_and_deprecates_val_loss_patience():
    schema = experiment_json_schema()
    training_ref = schema["properties"]["training"]["$ref"].split("/")[-1]
    training_properties = schema["$defs"][training_ref]["properties"]

    assert "warmup" in training_properties
    assert "auto_stop" in training_properties
    assert "debug_stop" in training_properties
    assert "val_loss_patience" in training_properties
    assert training_properties["val_loss_patience"].get("deprecated") is True


def test_experiment_template_schema_allows_placeholders():
    schema = experiment_template_json_schema()

    assert schema["title"] == "ExperimentTemplateConfig"
    assert "metadata" in schema["properties"]
    experiment_ref = schema["properties"]["experiment"]["$ref"].split("/")[-1]
    exp_name_schema = schema["$defs"][experiment_ref]["properties"]["exp_name"]
    assert "anyOf" in exp_name_schema
    assert any(item.get("pattern") for item in exp_name_schema["anyOf"])
