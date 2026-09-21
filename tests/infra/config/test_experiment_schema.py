from __future__ import annotations

import json
from pathlib import Path

import pytest

from lisai.config import load_yaml
import lisai.config.json_schema.experiment as experiment_schema_mod
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


def _registry_entry(
    *,
    inputs: list[str],
    targets: list[str] | None = None,
    data_format: str = "single",
    input_overrides: dict[str, str] | None = None,
) -> dict:
    input_overrides = input_overrides or {}
    targets = targets or []
    outputs = []
    for input_name in inputs:
        output = {"key": input_name or "image", "path": input_name, "role": "inp", "axes": "YX"}
        if input_name in input_overrides:
            output["data_format_override"] = input_overrides[input_name]
        outputs.append(output)
    outputs.extend({"key": target_name, "path": target_name, "role": "gt", "axes": "YX"} for target_name in targets)
    return {
        "data_format": data_format,
        "usage": "training",
        "for_training": True,
        "structure": {"recon": [*inputs, *targets]},
        "outputs": {"recon": outputs},
        "defaults": {"recon": {"input": inputs[0] if len(inputs) == 1 else None, "target": targets[0] if len(targets) == 1 else None}},
    }


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


def test_experiment_schema_adds_registry_dataset_suggestions_and_conditions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        experiment_schema_mod,
        "_load_dataset_registry_for_schema",
        lambda: {
            "demo": _registry_entry(
                inputs=["inp_single", "inp_mltpl_snr"],
                targets=["gt_avg"],
                data_format="mltpl_snr",
                input_overrides={"inp_single": "single"},
            )
        },
    )

    schema = experiment_json_schema()
    data_ref = schema["properties"]["data"]["$ref"].split("/")[-1]
    data_properties = schema["$defs"][data_ref]["properties"]
    dataset_name_schema = data_properties["dataset_name"]

    assert "demo" in dataset_name_schema["examples"]
    assert any(item.get("enum") == ["demo"] for item in dataset_name_schema["anyOf"])

    known_condition = next(
        condition
        for condition in schema["allOf"]
        if condition["if"]["properties"]["data"]["properties"]["dataset_name"].get("const") == "demo"
    )
    known_data_then = known_condition["then"]["properties"]["data"]

    assert known_data_then["properties"]["input"]["enum"] == ["inp_mltpl_snr", "inp_single"]
    assert known_data_then["properties"]["target"]["anyOf"][0]["enum"] == ["gt_avg"]
    assert known_data_then["properties"]["data_format"]["anyOf"][0]["enum"] == ["mltpl_snr", "single"]

    input_format_condition = next(
        condition
        for condition in known_data_then["allOf"]
        if condition.get("if", {}).get("properties", {}).get("input", {}).get("const") == "inp_single"
    )
    assert input_format_condition["then"]["properties"]["data_format"]["anyOf"][0]["enum"] == ["single"]


def test_experiment_schema_requires_unprepared_training_for_unknown_registry_dataset(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        experiment_schema_mod,
        "_load_dataset_registry_for_schema",
        lambda: {"demo": _registry_entry(inputs=["inp"])},
    )

    schema = experiment_json_schema()
    unknown_condition = schema["allOf"][-1]

    dataset_name_schema = unknown_condition["if"]["properties"]["data"]["properties"]["dataset_name"]
    assert dataset_name_schema == {"not": {"enum": ["demo"]}}
    assert unknown_condition["then"]["properties"]["data"]["required"] == ["prep_before"]
    assert unknown_condition["then"]["properties"]["data"]["properties"]["prep_before"] == {"const": False}


def test_experiment_template_schema_does_not_apply_unknown_dataset_rule_to_placeholders(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        experiment_schema_mod,
        "_load_dataset_registry_for_schema",
        lambda: {"demo": _registry_entry(inputs=["inp"])},
    )

    schema = experiment_template_json_schema()
    unknown_condition = schema["allOf"][-1]
    dataset_name_schema = unknown_condition["if"]["properties"]["data"]["properties"]["dataset_name"]

    assert dataset_name_schema["allOf"][0] == {"not": {"enum": ["demo"]}}
    assert dataset_name_schema["allOf"][1]["not"]["pattern"]
