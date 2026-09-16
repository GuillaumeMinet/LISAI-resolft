from __future__ import annotations

import yaml

import pytest

from lisai.promoted_models import catalog


SHA = "b" * 64


def _payload() -> bytes:
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "models": {
                "model-b": {
                    "task": "denoising_hdn",
                    "description": "B model",
                    "archive_sha256": SHA,
                    "download": {"url": "https://example.org/b.lisai.zip"},
                },
                "model-a": {
                    "task": "upsamp_single_frame",
                    "description": "A model",
                    "archive_sha256": SHA,
                    "download": {"url": "https://example.org/a.lisai.zip"},
                },
            },
        },
        sort_keys=False,
    ).encode("utf-8")


def test_list_models_fetches_catalog_and_sorts_by_name(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_catalog_bytes", lambda url, timeout: _payload())

    models = catalog.list_models(catalog_url="https://example.org/catalog.yaml")

    assert [model.name for model in models] == ["model-a", "model-b"]
    assert models[0].task == "upsamp_single_frame"


def test_get_model_returns_typed_catalog_entry(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_catalog_bytes", lambda url, timeout: _payload())

    model = catalog.get_model("model-b", catalog_url="https://example.org/catalog.yaml")

    assert model.name == "model-b"
    assert model.description == "B model"
    assert model.archive_sha256 == SHA


def test_get_model_requires_exact_catalog_name(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_catalog_bytes", lambda url, timeout: _payload())

    with pytest.raises(KeyError, match="lisai models catalog"):
        catalog.get_model("model", catalog_url="https://example.org/catalog.yaml")
