from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.evaluation import EvalSource, load_outputs_manifest
from lisai.runs.external import import_external_run
import lisai.runs.external.importer as importer_mod
import lisai.runs.external.discovery as discovery_mod


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def dataset_registry_path(self) -> Path:
        return self.root / "dataset_registry.yml"

    def dataset_preprocess_dir(self, *, dataset_name: str, data_type: str = "", usage: str = "training") -> Path:
        return self.root / usage / dataset_name / "preprocess" / data_type

    def external_run_dir(self, *, dataset_name: str, run_name: str) -> Path:
        return self.root / "training" / dataset_name / "external_runs" / run_name

    def training_datasets_root(self) -> Path:
        return self.root / "training"

    def dataset_external_runs_dir_from_dataset_dir(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / "external_runs"


def _write_registry(path: Path) -> None:
    save_yaml(
        {
            "train_ds": {
                "data_format": "single",
                "usage": "training",
                "for_training": True,
                "defaults": {"recon": {"input": "inp", "target": "gt", "eval_gt": "gt"}},
                "outputs": {
                    "recon": [
                        {"key": "inp", "path": "inp", "role": "inp", "axes": "YX"},
                        {"key": "gt", "path": "gt", "role": "gt", "axes": "YX"},
                    ]
                },
            },
            "eval_ds": {
                "data_format": "single",
                "usage": "evaluation",
                "for_training": False,
                "defaults": {"recon": {"input": "low", "target": None, "eval_gt": "high"}},
                "outputs": {
                    "recon": [
                        {"key": "inp", "path": "low", "role": "inp", "axes": "YX"},
                        {"key": "gt", "path": "high", "role": "gt", "axes": "YX"},
                        {"key": "conf", "path": "conf", "role": "aux", "axes": "YX"},
                    ]
                },
            },
        },
        path,
    )


def test_import_external_run_copies_artifacts_and_pairs_predictions_by_order(monkeypatch, tmp_path: Path):
    paths = FakePaths(tmp_path)
    _write_registry(paths.dataset_registry_path())
    monkeypatch.setattr(importer_mod, "Paths", lambda _settings: paths)

    eval_root = paths.dataset_preprocess_dir(dataset_name="eval_ds", data_type="recon", usage="evaluation")
    for subfolder in ("low", "high", "conf"):
        (eval_root / subfolder).mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(("a.tif", "b.tif")):
        imwrite(eval_root / "low" / name, np.full((4, 5), index, dtype=np.float32))
        imwrite(eval_root / "high" / name, np.full((4, 5), index + 10, dtype=np.float32))
        imwrite(eval_root / "conf" / name, np.full((4, 5), index + 20, dtype=np.float32))

    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    # Deliberately unrelated names: matching is count + natural file order.
    imwrite(pred_root / "prediction_10.tif", np.stack([np.zeros((4, 5)), np.full((4, 5), 22)]).astype(np.float32))
    imwrite(pred_root / "prediction_2.tif", np.stack([np.zeros((4, 5)), np.full((4, 5), 11)]).astype(np.float32))

    checkpoint = tmp_path / "weights.h5"
    checkpoint.write_bytes(b"weights")
    config = tmp_path / "n2v_config.json"
    config.write_text('{"epochs": 100}', encoding="utf-8")

    result = import_external_run(
        run_name="N2V_test",
        trained_on="train_ds",
        checkpoint_path=checkpoint,
        config_path=config,
        evaluation_data_path=pred_root,
        evaluated_on=EvalSource.dataset("eval_ds"),
        prediction_format="stack_inp_pred",
    )

    assert (result.run_dir / "checkpoints" / "weights.h5").read_bytes() == b"weights"
    assert (result.run_dir / "config" / "n2v_config.json").read_text(encoding="utf-8") == '{"epochs": 100}'
    metadata = load_yaml(result.run_dir / "external_run.yaml")
    assert metadata["kind"] == "external"
    assert metadata["trained_on"]["dataset"] == "train_ds"
    assert metadata["config"] == "config/n2v_config.json"

    # Natural ordering maps prediction_2 -> a and prediction_10 -> b.
    assert np.all(imread(result.evaluation_dir / "a_pred.tif") == 11)
    assert np.all(imread(result.evaluation_dir / "b_pred.tif") == 22)

    manifest = load_outputs_manifest(result.evaluation_dir)
    assert [item["input_id"] for item in manifest["items"]] == ["low/a.tif", "low/b.tif"]
    assert [item["gt_id"] for item in manifest["items"]] == ["high/a.tif", "high/b.tif"]
    assert all(set(item["outputs"]) == {"pred"} for item in manifest["items"])


def test_import_external_run_requires_explicit_snr_for_multi_snr(monkeypatch, tmp_path: Path):
    paths = FakePaths(tmp_path)
    save_yaml(
        {
            "train_ds": {
                "data_format": "mltpl_snr",
                "usage": "training",
                "for_training": True,
                "defaults": {"recon": {"input": "inp", "target": None, "eval_gt": None}},
                "outputs": {"recon": [{"key": "inp", "path": "inp", "role": "inp", "axes": "SYX"}]},
            },
            "eval_ds": {
                "data_format": "single",
                "usage": "evaluation",
                "for_training": False,
                "defaults": {"recon": {"input": "low", "target": None, "eval_gt": None}},
                "outputs": {"recon": [{"key": "inp", "path": "low", "role": "inp", "axes": "YX"}]},
            },
        },
        paths.dataset_registry_path(),
    )
    monkeypatch.setattr(importer_mod, "Paths", lambda _settings: paths)

    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    try:
        import_external_run(
            run_name="external",
            trained_on="train_ds",
            checkpoint_path=None,
            evaluation_data_path=pred_root,
            evaluated_on=EvalSource.dataset("eval_ds"),
        )
    except ValueError as exc:
        assert "Specify snr_idx" in str(exc)
    else:
        raise AssertionError("Expected multi-SNR training data to require snr_idx.")


def test_external_run_discovery(monkeypatch, tmp_path: Path):
    paths = FakePaths(tmp_path)
    run_dir = paths.external_run_dir(dataset_name="train_ds", run_name="N2V_test")
    run_dir.mkdir(parents=True)
    save_yaml(
        {
            "schema_version": 1,
            "kind": "external",
            "run_name": "N2V_test",
            "trained_on": {"dataset": "train_ds"},
            "uses_current_split": True,
            "imported_at": "2026-09-18T08:00:00Z",
        },
        run_dir / "external_run.yaml",
    )
    monkeypatch.setattr(discovery_mod, "Paths", lambda _settings: paths)

    scan = discovery_mod.scan_external_runs()
    assert scan.invalid == ()
    assert len(scan.runs) == 1
    assert scan.runs[0].dataset == "train_ds"
    assert scan.runs[0].name == "N2V_test"
