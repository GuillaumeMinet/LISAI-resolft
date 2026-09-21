from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

import lisai.cli as root_cli
import lisai.preprocess.cli as preprocess_cli
from lisai.config import save_yaml, settings
from lisai.preprocess.reporting import NoOpPreprocessReporter


class DummyPaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_registry_path(self) -> Path:
        return self.root / "dataset_registry.yml"

    def dataset_dump_dir(self, *, dataset_name: str, data_type: str = "", additional_subfolder: str = "", usage: str = "training") -> Path:
        return self.root / dataset_name / "dump" / data_type / additional_subfolder

    def dataset_preprocess_dir(self, *, dataset_name: str, data_type: str = "", usage: str = "training") -> Path:
        return self.root / dataset_name / "preprocess" / data_type

    def preprocess_log_path(self, *, dataset_name: str, data_type: str, usage: str = "training") -> Path:
        key = f"{data_type}_preprocess"
        return self.dataset_preprocess_dir(dataset_name=dataset_name, data_type=data_type, usage=usage) / settings.data_cfg.logs[key]

    def preprocessed_image_full_path(
        self,
        *,
        dataset_name: str,
        fmt: str,
        data_type: str = "",
        additional_subfolder: str = "",
        usage: str = "training",
        **kwargs,
    ) -> Path:
        filename = settings.get_data_filename(fmt=fmt, data_type=data_type, **kwargs)
        return self.dataset_preprocess_dir(dataset_name=dataset_name, data_type=data_type, usage=usage) / additional_subfolder / filename


def _write_single_source_dataset(root: Path, dataset_name: str, file_names: list[str]) -> None:
    dump_dir = root / dataset_name / "dump" / "recon"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for index, file_name in enumerate(file_names):
        image = np.full((8, 8), fill_value=index + 1, dtype=np.uint16)
        tifffile.imwrite(dump_dir / file_name, image)


def _write_config(path: Path, dataset_name: str) -> None:
    save_yaml(
        {
            "dataset_name": dataset_name,
            "pipeline": "single_recon",
            "data_type": "recon",
            "fmt": "single",
            "pipeline_cfg": {},
            "log": {"enabled": True},
            "split": {"enabled": False},
        },
        path,
    )


def test_resolve_config_path_supports_preprocess_short_name():
    repo_root = Path(__file__).resolve().parents[3]
    expected = (repo_root / "configs" / "preprocess" / "preprocess.yml").resolve()

    assert preprocess_cli.resolve_config_path("preprocess.yml") == expected


def test_resolve_config_path_supports_preprocess_short_name_without_extension():
    repo_root = Path(__file__).resolve().parents[3]
    expected = (repo_root / "configs" / "preprocess" / "preprocess.yml").resolve()

    assert preprocess_cli.resolve_config_path("preprocess") == expected


def test_resolve_config_path_lists_available_configs_when_missing():
    with pytest.raises(FileNotFoundError, match="Preprocess config not found: missing_preprocess_config") as exc_info:
        preprocess_cli.resolve_config_path("missing_preprocess_config")

    message = str(exc_info.value)
    assert "Available configs:" in message
    assert "preprocess.yml" in message


def test_run_preprocess_config_overwrites_existing_output_after_confirmation(tmp_path: Path):
    dataset_name = "CliOverwriteDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif"])
    config_path = tmp_path / "overwrite.yml"
    _write_config(config_path, dataset_name)

    preprocess_dir = tmp_path / dataset_name / "preprocess" / "recon"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    stale_file = preprocess_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    result = preprocess_cli.run_preprocess_config(
        config_path,
        paths=DummyPaths(tmp_path),
        reporter=NoOpPreprocessReporter(),
        input_fn=lambda _: "yes",
        interactive=True,
    )

    assert result.n_files == 1
    assert not stale_file.exists()
    assert (preprocess_dir / "c00.tif").exists()


def test_run_preprocess_config_aborts_when_overwrite_is_declined(tmp_path: Path):
    dataset_name = "CliAbortDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif"])
    config_path = tmp_path / "abort.yml"
    _write_config(config_path, dataset_name)

    preprocess_dir = tmp_path / dataset_name / "preprocess" / "recon"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    stale_file = preprocess_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    with pytest.raises(preprocess_cli.PreprocessAbortedError, match="left untouched"):
        preprocess_cli.run_preprocess_config(
            config_path,
            paths=DummyPaths(tmp_path),
            reporter=NoOpPreprocessReporter(),
            input_fn=lambda _: "no",
            interactive=True,
        )

    assert stale_file.exists()
    assert not (preprocess_dir / "c00.tif").exists()


def test_run_preprocess_config_requires_explicit_overwrite_in_non_interactive_mode(tmp_path: Path):
    dataset_name = "CliNonInteractiveDataset"
    _write_single_source_dataset(tmp_path, dataset_name, ["img_a.tif"])
    config_path = tmp_path / "noninteractive.yml"
    _write_config(config_path, dataset_name)

    preprocess_dir = tmp_path / dataset_name / "preprocess" / "recon"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    (preprocess_dir / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Existing preprocess output detected"):
        preprocess_cli.run_preprocess_config(
            config_path,
            paths=DummyPaths(tmp_path),
            reporter=NoOpPreprocessReporter(),
            interactive=False,
        )


def test_preprocess_cli_main_accepts_extensionless_config_name(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_preprocess_config(config_path):
        captured["config_path"] = Path(config_path)

    monkeypatch.setattr(preprocess_cli, "run_preprocess_config", fake_run_preprocess_config)

    exit_code = preprocess_cli.main(["preprocess"])

    assert exit_code == 0
    assert captured["config_path"].name == "preprocess.yml"


def test_root_cli_preprocess_dispatches_extensionless_config_to_preprocess(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_preprocess_config(config_path):
        captured["config_path"] = Path(config_path)

    monkeypatch.setattr(preprocess_cli, "run_preprocess_config", fake_run_preprocess_config)

    exit_code = root_cli.main(["preprocess", "preprocess"])

    assert exit_code == 0
    assert captured["config_path"].name == "preprocess.yml"
