from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import lisai.cli as root_cli
import lisai.config.catalog as catalog_mod
import lisai.config.io.resolver as resolver_mod
from lisai.config import cli as config_cli
from lisai.config.catalog import validate_training_config_dict, validate_training_template_dict
from lisai.config.io.yaml import load_yaml


def _registry_entry(
    *,
    inputs: list[str] | None = None,
    targets: list[str] | None = None,
    input_default: str | None = None,
    target_default: str | None = None,
    data_format: str = "single",
    input_overrides: dict[str, str] | None = None,
) -> dict:
    inputs = inputs if inputs is not None else ([] if input_default is None else [input_default])
    targets = targets if targets is not None else ([] if target_default is None else [target_default])
    input_overrides = input_overrides or {}
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
        "defaults": {"recon": {"input": input_default, "target": target_default, "eval_gt": target_default}},
    }


def test_configs_list_filters_presets(capsys: pytest.CaptureFixture[str]):
    exit_code = config_cli.main(["list", "--kind", "preset"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "presets/denoising_hdn_unsup.yml" in captured.out
    assert "presets/upsamp_single_frame.yml" in captured.out


def test_root_cli_configs_dispatches(capsys: pytest.CaptureFixture[str]):
    exit_code = root_cli.main(["configs", "list", "--kind", "template"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "templates/base_training.yml" in captured.out


def test_template_validation_accepts_placeholders():
    cfg = load_yaml(Path("configs/training/templates/base_training.yml"))

    validate_training_template_dict(cfg)


def test_training_validation_strips_metadata_from_preset():
    cfg = load_yaml(Path("configs/training/presets/denoising_hdn_unsup.yml"))

    validated = validate_training_config_dict(cfg)

    assert validated.experiment.task.name == "denoising_hdn"


def test_configs_new_hdn_preset_patches_dataset_and_keeps_task_name(tmp_path: Path):
    output = tmp_path / "hdn_sup.yml"

    exit_code = config_cli.main(
        [
            "new",
            "denoising_hdn_sup",
            "--name",
            "hdn_sup",
            "--dataset",
            "vim_fixed",
            "--input",
            "inp_single",
            "--target",
            "gt_avg",
            "--betaKL",
            "0.1",
            "--output",
            str(output),
        ]
    )

    cfg = load_yaml(output)
    text = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert text.startswith("# yaml-language-server: $schema=../../schema/experiment.schema.json")
    assert "# General experiment settings" in text
    assert "# Dataset / DataLoader configuration" in text
    assert "metadata" not in cfg
    assert cfg["experiment"]["exp_name"] == "hdn_sup"
    assert cfg["experiment"]["task"]["name"] == "denoising_hdn"
    assert cfg["experiment"]["task"]["supervised"] is True
    assert cfg["experiment"]["task"]["betaKL"] == 0.1
    assert cfg["data"]["dataset_name"] == "vim_fixed"
    assert cfg["data"]["input"] == "inp_single"
    assert cfg["data"]["target"] == "gt_avg"


def test_configs_new_accepts_empty_input_flag(tmp_path: Path):
    output = tmp_path / "empty_input.yml"

    exit_code = config_cli.main(
        [
            "new",
            "denoising_hdn_unsup",
            "--name",
            "empty_input",
            "--dataset",
            "vim_fixed",
            "--input",
            "--output",
            str(output),
        ]
    )

    cfg = load_yaml(output)
    assert exit_code == 0
    assert cfg["data"]["input"] == ""


def test_configs_new_denoising_care_loss_stays_task_level(tmp_path: Path):
    output = tmp_path / "care.yml"

    config_cli.main(
        [
            "new",
            "denoising_care",
            "--name",
            "care_mse",
            "--dataset",
            "vim_fixed",
            "--input",
            "inp_single",
            "--target",
            "gt_avg",
            "--loss",
            "MSE",
            "--output",
            str(output),
        ]
    )

    cfg = load_yaml(output)
    assert cfg["experiment"]["task"]["loss"] == "MSE"
    assert "loss_function" not in cfg


def test_configs_new_custom_starts_from_base_template(tmp_path: Path):
    output = tmp_path / "scratch.yml"

    exit_code = config_cli.main(
        [
            "new",
            "--custom",
            "--name",
            "scratch",
            "--output",
            str(output),
        ]
    )

    cfg = load_yaml(output)
    text = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert text.startswith("# yaml-language-server: $schema=../../schema/experiment.schema.json")
    assert "metadata" not in cfg
    assert cfg["experiment"]["exp_name"] == "scratch"
    assert cfg["experiment"]["task"]["name"] == "custom"
    assert cfg["data"]["dataset_name"] == "CHANGEME"
    assert cfg["data"]["input"] == "CHANGEME"
    assert cfg["model"]["architecture"] == "CHANGEME"


def test_configs_new_can_scaffold_with_missing_editable_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "scaffold.yml"

    exit_code = config_cli.main(
        [
            "new",
            "denoising_hdn_unsup",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    cfg = load_yaml(output)
    assert exit_code == 0
    assert cfg["experiment"]["exp_name"] == "CHANGEME"
    assert cfg["data"]["dataset_name"] == "CHANGEME"
    assert cfg["data"]["input"] == "CHANGEME"
    assert "Warning: experiment.exp_name was left as CHANGEME." in captured.out
    assert "Warning: data.dataset_name was left as CHANGEME." in captured.out
    assert "Warning: data.input was left as CHANGEME." in captured.out
    assert "Warning: experiment.task.betaKL kept from the preset." in captured.out


def test_configs_new_fills_registry_defaults_for_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "registry_defaults.yml"
    monkeypatch.setattr(
        catalog_mod,
        "load_dataset_registry",
        lambda path: {
            "demo": _registry_entry(
                inputs=["inp_single", "inp_mltpl_snr"],
                targets=["gt_avg"],
                input_default="inp_single",
                target_default="gt_avg",
                data_format="mltpl_snr",
                input_overrides={"inp_single": "single"},
            )
        },
    )

    exit_code = config_cli.main(
        [
            "new",
            "denoising_hdn_sup",
            "--name",
            "registry_defaults",
            "--dataset",
            "demo",
            "--output",
            str(output),
        ]
    )

    cfg = load_yaml(output)
    assert exit_code == 0
    assert cfg["data"]["dataset_name"] == "demo"
    assert cfg["data"]["input"] == "inp_single"
    assert cfg["data"]["target"] == "gt_avg"
    assert cfg["data"]["data_format"] == "single"


def test_configs_new_keeps_cli_dataset_fields_over_registry_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "registry_cli_priority.yml"
    monkeypatch.setattr(
        catalog_mod,
        "load_dataset_registry",
        lambda path: {
            "demo": _registry_entry(
                inputs=["inp_registry", "inp_cli"],
                targets=["gt_registry", "gt_cli"],
                input_default="inp_registry",
                target_default="gt_registry",
                data_format="mltpl_snr",
                input_overrides={"inp_cli": "single"},
            )
        },
    )

    exit_code = config_cli.main(
        [
            "new",
            "denoising_hdn_sup",
            "--name",
            "registry_cli_priority",
            "--dataset",
            "demo",
            "--input",
            "inp_cli",
            "--target",
            "gt_cli",
            "--output",
            str(output),
        ]
    )

    cfg = load_yaml(output)
    assert exit_code == 0
    assert cfg["data"]["input"] == "inp_cli"
    assert cfg["data"]["target"] == "gt_cli"
    assert cfg["data"]["data_format"] == "single"


def test_configs_new_leaves_ambiguous_registry_choices_as_changeme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "registry_ambiguous.yml"
    monkeypatch.setattr(
        catalog_mod,
        "load_dataset_registry",
        lambda path: {
            "demo": _registry_entry(
                inputs=["inp_a", "inp_b"],
                targets=["gt_a", "gt_b"],
                input_default=None,
                target_default=None,
            )
        },
    )

    exit_code = config_cli.main(
        [
            "new",
            "denoising_hdn_sup",
            "--name",
            "registry_ambiguous",
            "--dataset",
            "demo",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    cfg = load_yaml(output)
    assert exit_code == 0
    assert cfg["data"]["input"] == "CHANGEME"
    assert cfg["data"]["target"] == "CHANGEME"
    assert "Warning: data.input was left as CHANGEME." in captured.out
    assert "Warning: data.target was left as CHANGEME for supervised denoising_hdn." in captured.out


def test_configs_validate_rejects_scaffolded_changeme_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "scaffold.yml"
    config_cli.main(["new", "denoising_hdn_unsup", "--output", str(output)])

    with pytest.raises(SystemExit) as exc_info:
        config_cli.main(["validate", str(output)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Training config still contains CHANGEME placeholder" in captured.err
    assert "experiment.exp_name" in captured.err
    assert "data.dataset_name" in captured.err


def test_configs_resolve_strips_metadata(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        resolver_mod,
        "load_dataset_registry",
        lambda path: {
            "replace_me_dataset": {
                "data_format": "single",
                "usage": "training",
                "for_training": True,
                "structure": {"recon": ["replace_me_input"]},
                "outputs": {
                    "recon": [
                        {"key": "replace_me_input", "path": "replace_me_input", "role": "inp", "axes": "YX"}
                    ]
                },
                "defaults": {"recon": {"input": "replace_me_input", "target": None, "eval_gt": None}},
            }
        },
    )

    exit_code = config_cli.main(["resolve", "presets/denoising_hdn_unsup"])

    captured = capsys.readouterr()
    resolved = yaml.safe_load(captured.out)
    assert exit_code == 0
    assert "metadata" not in resolved
    assert resolved["experiment"]["task"]["name"] == "denoising_hdn"
