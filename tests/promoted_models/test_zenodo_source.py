from __future__ import annotations

import pytest

from lisai.promoted_models.catalog_schema import ZenodoDownloadSource
from lisai.promoted_models.sources import zenodo


def test_zenodo_source_resolves_exact_filename(monkeypatch):
    monkeypatch.setattr(
        zenodo,
        "_fetch_record",
        lambda record_id, timeout: {
            "files": [
                {
                    "key": "demo-model.lisai.zip",
                    "links": {"self": "https://zenodo.org/api/files/example/demo-model.lisai.zip"},
                }
            ]
        },
    )
    source = ZenodoDownloadSource(
        type="zenodo",
        record_id="12345678",
        filename="demo-model.lisai.zip",
    )

    url = zenodo.resolve_download_url(source)

    assert url.endswith("demo-model.lisai.zip")


def test_zenodo_source_rejects_missing_file(monkeypatch):
    monkeypatch.setattr(
        zenodo,
        "_fetch_record",
        lambda record_id, timeout: {"files": []},
    )
    source = ZenodoDownloadSource(
        type="zenodo",
        record_id="12345678",
        filename="demo-model.lisai.zip",
    )

    with pytest.raises(zenodo.ZenodoSourceError, match="does not contain"):
        zenodo.resolve_download_url(source)
