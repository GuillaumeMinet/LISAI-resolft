from __future__ import annotations

from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.data.cli as dataset_cli
from lisai.config.io.yaml import save_yaml


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_registry_path(self) -> Path:
        return self.root / "dataset_registry.yml"

    def dataset_dir(
        self, *, dataset_name: str, data_subfolder: str = "", usage: str = "training"
    ) -> Path:
        return self.root / usage / dataset_name / data_subfolder


def _write_registry(root: Path) -> None:
    save_yaml(
        {
            "Gag_noisy_single": {
                "data_format": "single",
                "usage": "training",
                "size": {
                    "recon": {
                        "n_files": 8,
                    }
                },
                "split": {
                    "recon": {
                        "counts": {"train": 6, "val": 2, "test": 0},
                    }
                },
                "outputs": {
                    "recon": [
                        {"key": "main", "path": "", "role": "inp", "axes": "YX"},
                    ]
                },
                "defaults": {
                    "recon": {"input": "", "target": None, "eval_gt": None},
                },
            },
            "Gag_noisy_timelapses": {
                "data_format": "timelapse",
                "usage": "training",
                "size": {
                    "recon": {
                        "n_files": 8,
                        "n_frames": 338,
                        "timepoints": {"min": 31, "max": 50},
                    }
                },
                "split": {
                    "recon": {
                        "counts": {"train": 6, "val": 1, "test": 1},
                    }
                },
                "outputs": {
                    "recon": [
                        {"key": "main", "path": "", "role": "inp", "axes": "TYX"},
                    ]
                },
                "defaults": {
                    "recon": {"input": "", "target": None, "eval_gt": None},
                },
            },
            "AAA_eval": {
                "data_format": "single",
                "usage": "evaluation",
                "size": {"recon": {"n_files": 2}},
                "outputs": {
                    "recon": [
                        {"key": "main", "path": "", "role": "inp", "axes": "YX"},
                    ]
                },
                "defaults": {
                    "recon": {"input": "", "target": None, "eval_gt": None},
                },
            },
            "vim_fixed": {
                "data_format": "mltpl_snr",
                "usage": "training",
                "size": {
                    "recon": {
                        "n_files": 25,
                        "n_frames": 147,
                        "snr_levels": {"min": 5, "max": 6},
                    }
                },
                "split": {
                    "recon": {
                        "counts": {"train": 21, "val": 1, "test": 3},
                    }
                },
                "outputs": {
                    "recon": [
                        {"key": "inp_mltpl_snr", "path": "inp_mltpl_snr", "role": "inp", "axes": "TYX"},
                        {"key": "inp_single", "path": "inp_single", "role": "inp", "axes": "YX"},
                        {"key": "gt_snr0", "path": "gt_snr0", "role": "gt", "axes": "YX"},
                        {"key": "gt_avg", "path": "gt_avg", "role": "gt", "axes": "YX"},
                    ]
                },
                "defaults": {
                    "recon": {"input": None, "target": None, "eval_gt": None},
                },
            },
        },
        root / "dataset_registry.yml",
    )


def _patch_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(dataset_cli, "Paths", lambda _settings: FakePaths(root))


def test_datasets_list_renders_compact_table(monkeypatch, tmp_path: Path, capsys):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)

    exit_code = root_cli.main(["datasets", "list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "name" in captured.out
    assert "files" in captured.out
    assert "frames" in captured.out
    assert "Gag_noisy_single" in captured.out
    gag_single_line = next(
        line for line in captured.out.splitlines() if line.startswith("Gag_noisy_single")
    )
    assert gag_single_line.split()[:6] == [
        "Gag_noisy_single", "training", "single", "recon", "8", "8"
    ]
    assert "Gag_noisy_timelapses" in captured.out
    assert "timelapse" in captured.out
    assert "338" in captured.out
    assert "6/1/1" in captured.out
    assert "t=31-50" in captured.out
    assert "in=<root>" in captured.out
    assert "vim_fixed" in captured.out
    assert "147" in captured.out
    assert "snr=5-6" in captured.out


def test_datasets_show_renders_dataset_details(monkeypatch, tmp_path: Path, capsys):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)

    exit_code = root_cli.main(["datasets", "show", "vim_fixed"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dataset: vim_fixed" in captured.out
    assert f"Path: {(tmp_path / 'training' / 'vim_fixed').resolve()}" in captured.out
    assert "Format: mltpl_snr" in captured.out
    assert "snr_levels: 5-6" in captured.out
    assert "split: train=21 val=1 test=3" in captured.out
    assert "inp_mltpl_snr" in captured.out
    assert "eval_gt: null" in captured.out


def test_datasets_open_uses_dataset_directory(monkeypatch, tmp_path: Path):
    _write_registry(tmp_path)
    (tmp_path / "training" / "vim_fixed").mkdir(parents=True)
    _patch_paths(monkeypatch, tmp_path)
    opened = {}

    def fake_open(path: Path) -> bool:
        opened["path"] = path
        return True

    monkeypatch.setattr(dataset_cli, "_try_open_path", fake_open)

    exit_code = root_cli.main(["datasets", "open", "vim_fixed"])

    assert exit_code == 0
    assert opened["path"] == (tmp_path / "training" / "vim_fixed").resolve()


def test_datasets_open_prints_path_when_explorer_launch_fails(monkeypatch, tmp_path: Path, capsys):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(dataset_cli, "_try_open_path", lambda _path: False)

    exit_code = root_cli.main(["datasets", "open", "vim_fixed"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == str((tmp_path / "training" / "vim_fixed").resolve())


def test_datasets_list_groups_training_before_evaluation(monkeypatch, tmp_path: Path, capsys):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)

    exit_code = root_cli.main(["datasets", "list"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.index("vim_fixed") < captured.out.index("AAA_eval")


def test_datasets_list_filters_by_usage(monkeypatch, tmp_path: Path, capsys):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)

    assert root_cli.main(["datasets", "list", "--training"]) == 0
    training = capsys.readouterr().out
    assert "vim_fixed" in training
    assert "AAA_eval" not in training

    assert root_cli.main(["datasets", "list", "--evaluation"]) == 0
    evaluation = capsys.readouterr().out
    assert "AAA_eval" in evaluation
    assert "vim_fixed" not in evaluation


def test_datasets_list_usage_filters_are_mutually_exclusive(monkeypatch, tmp_path: Path):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        root_cli.main(["datasets", "list", "--training", "--evaluation"])

    assert exc_info.value.code == 2


def test_datasets_show_unknown_dataset_exits(monkeypatch, tmp_path: Path, capsys):
    _write_registry(tmp_path)
    _patch_paths(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        root_cli.main(["datasets", "show", "missing"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Unknown dataset 'missing'" in captured.err
    assert "vim_fixed" in captured.err
