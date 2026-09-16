from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.data.cli as dataset_cli
import lisai.data.rename as dataset_rename
from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.data.dataset_registry import load_dataset_registry
from lisai.runs.schema import RunMetadata


class InteractiveInput(StringIO):
    def isatty(self) -> bool:
        return True


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def datasets_root(self) -> Path:
        return self.root / "datasets"

    def dataset_registry_path(self) -> Path:
        return self.datasets_root() / "dataset_registry.yml"

    def dataset_dir(
        self, *, dataset_name: str, usage: str = "training", data_subfolder: str = ""
    ) -> Path:
        return self.datasets_root() / usage / dataset_name / data_subfolder

    def dataset_runs_dir_from_dataset_dir(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / "runs"

    def cfg_train_path(self, *, run_dir: str | Path) -> Path:
        return Path(run_dir) / "config_train.yaml"

    def split_manifest_path(self, *, run_dir: str | Path) -> Path:
        return Path(run_dir) / "split_manifest.json"


def _write_registry(paths: FakePaths, *, usage: str = "training") -> None:
    paths.dataset_registry_path().parent.mkdir(parents=True, exist_ok=True)
    save_yaml(
        {
            "OldDataset": {
                "usage": usage,
                "data_format": "single",
                "defaults": {"recon": {"input": "", "target": None, "eval_gt": None}},
            }
        },
        paths.dataset_registry_path(),
    )


def _metadata_payload(*, dataset: str, path: str, status: str = "completed") -> dict:
    return {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": "demo",
        "run_index": 1,
        "dataset": dataset,
        "model_subfolder": "Upsamp",
        "kept": False,
        "status": status,
        "closed_cleanly": status not in {"running", "pause_requested", "paused", "resuming"},
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": (
            None
            if status in {"running", "pause_requested", "paused", "resuming"}
            else "2026-03-20T10:20:00Z"
        ),
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": path,
        "group_path": None,
    }


def _write_run(
    paths: FakePaths,
    *,
    archived: bool = False,
    status: str = "completed",
) -> Path:
    dataset_dir = paths.dataset_dir(dataset_name="OldDataset", usage="training")
    run_dir = dataset_dir / "runs" / "Upsamp"
    if archived:
        run_dir = run_dir / "_archive" / "demo_01_archived_20260914-100000Z"
    else:
        run_dir = run_dir / "demo_01"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = _metadata_payload(
        dataset="OldDataset",
        path="datasets/training/OldDataset/runs/Upsamp/demo_01",
        status=status,
    )
    (run_dir / ".lisai_run_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return run_dir


def _write_active_run_artifacts(run_dir: Path) -> None:
    save_yaml(
        {
            "data": {"dataset_name": "OldDataset", "input": "recon"},
            "model": {"architecture": "UNet"},
        },
        run_dir / "config_train.yaml",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "dataset_name": "OldDataset",
                "splits": {"train": [], "val": [], "test": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_training_dataset_rename_updates_live_metadata_and_archived_metadata(tmp_path: Path):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    active = _write_run(paths)
    archived = _write_run(paths, archived=True)
    _write_active_run_artifacts(active)
    _write_active_run_artifacts(archived)

    plan = dataset_rename.build_dataset_rename_plan(
        "OldDataset", "NewDataset", paths=paths
    )

    assert plan.active_run_metadata_count == 1
    assert plan.training_config_count == 1
    assert plan.split_manifest_count == 1
    assert plan.archived_run_metadata_count == 1

    dataset_rename.apply_dataset_rename(plan)

    old_dir = paths.dataset_dir(dataset_name="OldDataset", usage="training")
    new_dir = paths.dataset_dir(dataset_name="NewDataset", usage="training")
    assert not old_dir.exists()
    assert new_dir.is_dir()

    registry = load_dataset_registry(paths.dataset_registry_path())
    assert "OldDataset" not in registry
    assert registry["NewDataset"]["usage"] == "training"

    new_active = new_dir / active.relative_to(old_dir)
    active_meta = RunMetadata.model_validate_json(
        (new_active / ".lisai_run_meta.json").read_text(encoding="utf-8")
    )
    assert active_meta.dataset == "NewDataset"
    assert active_meta.path == "datasets/training/NewDataset/runs/Upsamp/demo_01"
    assert load_yaml(new_active / "config_train.yaml")["data"]["dataset_name"] == "NewDataset"
    manifest = json.loads((new_active / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "NewDataset"

    new_archived = new_dir / archived.relative_to(old_dir)
    archived_meta = RunMetadata.model_validate_json(
        (new_archived / ".lisai_run_meta.json").read_text(encoding="utf-8")
    )
    assert archived_meta.dataset == "NewDataset"
    assert archived_meta.path.startswith("datasets/training/NewDataset/")
    # Archived configs are historical snapshots and are deliberately not rewritten.
    assert load_yaml(new_archived / "config_train.yaml")["data"]["dataset_name"] == "OldDataset"


def test_evaluation_dataset_rename_only_moves_folder_and_registry(tmp_path: Path):
    paths = FakePaths(tmp_path)
    _write_registry(paths, usage="evaluation")
    old_dir = paths.dataset_dir(dataset_name="OldDataset", usage="evaluation")
    old_dir.mkdir(parents=True)
    (old_dir / "image.tif").write_bytes(b"demo")

    plan = dataset_rename.build_dataset_rename_plan(
        "OldDataset", "NewDataset", paths=paths
    )
    assert plan.rewrites == ()

    dataset_rename.apply_dataset_rename(plan)

    new_dir = paths.dataset_dir(dataset_name="NewDataset", usage="evaluation")
    assert (new_dir / "image.tif").read_bytes() == b"demo"
    assert "NewDataset" in load_dataset_registry(paths.dataset_registry_path())


def test_dataset_rename_refuses_non_terminal_run(tmp_path: Path):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    _write_run(paths, status="running")

    with pytest.raises(dataset_rename.DatasetRenameError, match="non-terminal training run"):
        dataset_rename.build_dataset_rename_plan("OldDataset", "NewDataset", paths=paths)


def test_dataset_rename_preflight_refuses_existing_destination(tmp_path: Path):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    paths.dataset_dir(dataset_name="OldDataset", usage="training").mkdir(parents=True)
    paths.dataset_dir(dataset_name="NewDataset", usage="training").mkdir(parents=True)

    with pytest.raises(dataset_rename.DatasetRenameError, match="already exists"):
        dataset_rename.build_dataset_rename_plan("OldDataset", "NewDataset", paths=paths)


def test_dataset_rename_rolls_back_folder_registry_and_metadata_on_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    active = _write_run(paths)
    _write_active_run_artifacts(active)
    original_registry = paths.dataset_registry_path().read_bytes()
    original_metadata = (active / ".lisai_run_meta.json").read_bytes()

    plan = dataset_rename.build_dataset_rename_plan(
        "OldDataset", "NewDataset", paths=paths
    )
    real_write = dataset_rename._atomic_write_bytes
    failed = False

    def fail_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if path.name == "split_manifest.json" and not failed:
            failed = True
            raise OSError("simulated write failure")
        real_write(path, data)

    monkeypatch.setattr(dataset_rename, "_atomic_write_bytes", fail_once)

    with pytest.raises(dataset_rename.DatasetRenameError, match="rollback was attempted"):
        dataset_rename.apply_dataset_rename(plan)

    old_dir = paths.dataset_dir(dataset_name="OldDataset", usage="training")
    new_dir = paths.dataset_dir(dataset_name="NewDataset", usage="training")
    assert old_dir.is_dir()
    assert not new_dir.exists()
    assert paths.dataset_registry_path().read_bytes() == original_registry
    restored_metadata = old_dir / active.relative_to(old_dir) / ".lisai_run_meta.json"
    assert restored_metadata.read_bytes() == original_metadata


def test_datasets_rename_cli_requires_confirmation_and_warns_about_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    paths.dataset_dir(dataset_name="OldDataset", usage="training").mkdir(parents=True)
    monkeypatch.setattr(dataset_cli, "Paths", lambda _settings: paths)
    monkeypatch.setattr("sys.stdin", StringIO("n\n"))

    assert root_cli.main(["datasets", "rename", "OldDataset", "NewDataset"]) == 0
    output = capsys.readouterr().out

    assert "Dataset rename" in output
    assert "OldDataset  ->  NewDataset" in output
    assert "TensorBoard logs" in output
    assert "Proceed with dataset rename? [y/N]" in output
    assert "Dataset rename cancelled." in output
    assert paths.dataset_dir(dataset_name="OldDataset", usage="training").exists()
    assert not paths.dataset_dir(dataset_name="NewDataset", usage="training").exists()


def test_datasets_rename_cli_confirmation_applies_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    paths.dataset_dir(dataset_name="OldDataset", usage="training").mkdir(parents=True)
    monkeypatch.setattr(dataset_cli, "Paths", lambda _settings: paths)
    monkeypatch.setattr("sys.stdin", StringIO("yes\n"))

    assert root_cli.main(["datasets", "rename", "OldDataset", "NewDataset"]) == 0
    output = capsys.readouterr().out

    assert "Renamed dataset: OldDataset -> NewDataset" in output
    assert not paths.dataset_dir(dataset_name="OldDataset", usage="training").exists()
    assert paths.dataset_dir(dataset_name="NewDataset", usage="training").exists()


def test_datasets_rename_has_no_yes_bypass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    paths.dataset_dir(dataset_name="OldDataset", usage="training").mkdir(parents=True)
    monkeypatch.setattr(dataset_cli, "Paths", lambda _settings: paths)

    with pytest.raises(SystemExit) as exc_info:
        root_cli.main(["datasets", "rename", "OldDataset", "NewDataset", "--yes"])

    assert exc_info.value.code == 2


def test_datasets_rename_accepts_unique_partial_source_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
):
    paths = FakePaths(tmp_path)
    _write_registry(paths)
    paths.dataset_dir(dataset_name="OldDataset", usage="training").mkdir(parents=True)
    monkeypatch.setattr(dataset_cli, "Paths", lambda _settings: paths)
    monkeypatch.setattr("sys.stdin", InteractiveInput("yes\nyes\n"))

    assert root_cli.main(["datasets", "rename", "Old", "NewDataset"]) == 0
    output = capsys.readouterr().out

    assert "OldDataset  ->  NewDataset" in output
    assert "Renamed dataset: OldDataset -> NewDataset" in output
    assert not paths.dataset_dir(dataset_name="OldDataset", usage="training").exists()
    assert paths.dataset_dir(dataset_name="NewDataset", usage="training").exists()
