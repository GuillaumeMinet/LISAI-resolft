from __future__ import annotations

import json

import pytest

from lisai.data import catalog


SHA = "a" * 64


def _payload() -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "datasets": {
                "train_a": {
                    "usage": "training",
                    "install_path": "datasets/training",
                    "archives": [
                        {
                            "filename": "train_a.zip",
                            "size_bytes": 123,
                            "sha256": SHA,
                        }
                    ],
                    "registry_entry": {
                        "usage": "training",
                        "data_format": "single",
                        "description": "Training A",
                    },
                    "dependencies": {
                        "evaluation_datasets": ["eval_a"],
                        "noise_models": ["noise_a"],
                    },
                },
                "eval_a": {
                    "usage": "evaluation",
                    "install_path": "datasets/evaluation",
                    "archives": [
                        {
                            "filename": "eval_a.zip",
                            "size_bytes": 45,
                            "sha256": SHA,
                        }
                    ],
                    "registry_entry": {
                        "usage": "evaluation",
                        "data_format": "single",
                    },
                    "dependencies": {
                        "evaluation_datasets": [],
                        "noise_models": [],
                    },
                },
            },
            "shared": {
                "noise_models": {
                    "archive": {
                        "filename": "noise_models.zip",
                        "size_bytes": 10,
                        "sha256": SHA,
                    },
                    "available": ["noise_a"],
                }
            },
        }
    ).encode("utf-8")


def test_dataset_catalog_loads_builder_schema(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_catalog_bytes", lambda url, timeout: _payload())

    loaded = catalog.load_catalog(catalog_url="https://example.org/dataset_catalog.json")

    assert list(loaded.datasets) == ["train_a", "eval_a"]
    assert loaded.datasets["train_a"].dependencies.evaluation_datasets == ["eval_a"]
    assert catalog.dataset_download_size(loaded.datasets["train_a"]) == 123
    assert loaded.shared is not None
    assert loaded.shared.noise_models is not None
    assert loaded.shared.noise_models.available == ["noise_a"]


def test_dataset_catalog_rejects_wrong_install_path(monkeypatch):
    payload = json.loads(_payload())
    payload["datasets"]["train_a"]["install_path"] = "datasets/evaluation"
    monkeypatch.setattr(
        catalog,
        "_fetch_catalog_bytes",
        lambda url, timeout: json.dumps(payload).encode("utf-8"),
    )

    with pytest.raises(ValueError, match="expected 'datasets/training'"):
        catalog.load_catalog(catalog_url="https://example.org/dataset_catalog.json")


def test_dataset_catalog_loads_local_file(tmp_path):
    path = tmp_path / "dataset_catalog.json"
    path.write_bytes(_payload())

    loaded = catalog.load_catalog_file(path)

    assert set(loaded.datasets) == {"train_a", "eval_a"}
    assert loaded.datasets["train_a"].dependencies.noise_models == ["noise_a"]
