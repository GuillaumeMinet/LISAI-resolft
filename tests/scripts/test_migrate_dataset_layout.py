from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from lisai.config.io.yaml import save_yaml
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.schema import RunMetadata


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_dataset_layout.py"
spec = importlib.util.spec_from_file_location("migrate_dataset_layout", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
migration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = migration
spec.loader.exec_module(migration)


class FakePaths:
    def __init__(self, root: Path):
        self.root = root
        self.mapping = {"training": "train_bucket", "evaluation": "eval_bucket"}

    def datasets_root(self) -> Path:
        return self.root / "datasets"

    def dataset_registry_path(self) -> Path:
        return self.datasets_root() / "dataset_registry.yml"

    def dataset_usage_subfolder(self, usage: str) -> str:
        return self.mapping[usage]

    def dataset_dir(self, *, dataset_name: str, usage: str = "training", data_subfolder: str = "") -> Path:
        return self.datasets_root() / self.mapping[usage] / dataset_name / data_subfolder

    def dataset_runs_dir_from_dataset_dir(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / "models"


def _write_run(run_dir: Path, *, dataset: str, path: str) -> None:
    run_name, run_index = parse_run_dir_name(run_dir.name)
    payload = {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": "HDN",
        "status": "completed",
        "closed_cleanly": True,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:20:00Z",
        "ended_at": "2026-03-20T10:20:00Z",
        "last_heartbeat_at": "2026-03-20T10:20:00Z",
        "last_epoch": 10,
        "max_epoch": 10,
        "best_val_loss": 0.1,
        "path": path,
        "group_path": None,
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_migration_moves_by_registry_usage_and_rewrites_run_paths(tmp_path: Path, capsys):
    paths = FakePaths(tmp_path)
    datasets_root = paths.datasets_root()
    datasets_root.mkdir(parents=True)
    save_yaml(
        {
            "TrainSet": {"usage": "training", "data_format": "single"},
            "EvalSet": {"usage": "evaluation", "data_format": "single"},
        },
        paths.dataset_registry_path(),
    )

    old_train = datasets_root / "TrainSet"
    old_eval = datasets_root / "EvalSet"
    old_eval.mkdir()
    run_dir = old_train / "models" / "HDN" / "demo_00"
    _write_run(
        run_dir,
        dataset="TrainSet",
        path="datasets/TrainSet/models/HDN/demo_00",
    )

    assert migration.migrate(dry_run=True, paths=paths) == 0
    dry_run_output = capsys.readouterr().out
    assert "datasets/TrainSet" in dry_run_output
    assert "datasets/train_bucket/TrainSet" in dry_run_output
    assert "datasets/train_bucket/TrainSet/models/HDN/demo_00" in dry_run_output
    assert old_train.exists()
    assert old_eval.exists()

    assert migration.migrate(dry_run=False, paths=paths) == 0
    capsys.readouterr()

    new_train = paths.dataset_dir(dataset_name="TrainSet", usage="training")
    new_eval = paths.dataset_dir(dataset_name="EvalSet", usage="evaluation")
    assert new_train.is_dir()
    assert new_eval.is_dir()
    assert not old_train.exists()
    assert not old_eval.exists()

    metadata = read_run_metadata(new_train / "models" / "HDN" / "demo_00")
    assert metadata.path == "datasets/train_bucket/TrainSet/models/HDN/demo_00"


def test_migration_aborts_before_moving_when_destination_conflicts(tmp_path: Path):
    paths = FakePaths(tmp_path)
    datasets_root = paths.datasets_root()
    datasets_root.mkdir(parents=True)
    save_yaml(
        {"TrainSet": {"usage": "training", "data_format": "single"}},
        paths.dataset_registry_path(),
    )

    (datasets_root / "TrainSet").mkdir()
    paths.dataset_dir(dataset_name="TrainSet", usage="training").mkdir(parents=True)

    assert migration.migrate(dry_run=False, paths=paths) == 2
    assert (datasets_root / "TrainSet").is_dir()
