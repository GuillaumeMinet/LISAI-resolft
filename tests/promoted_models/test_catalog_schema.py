from __future__ import annotations

import pytest
from pydantic import ValidationError

from lisai.promoted_models.catalog_schema import DownloadableModelCatalog


SHA = "a" * 64


def _entry(download: dict) -> dict:
    return {
        "task": "upsamp_multiframes",
        "description": "Five-frame upsampling model.",
        "archive_sha256": SHA,
        "download": download,
    }


def test_catalog_accepts_direct_url_and_zenodo_source():
    catalog = DownloadableModelCatalog.model_validate(
        {
            "schema_version": 1,
            "models": {
                "direct-model": _entry({"url": "https://example.org/direct.lisai.zip"}),
                "zenodo-model": _entry(
                    {
                        "source": {
                            "type": "zenodo",
                            "record_id": "12345678",
                            "filename": "zenodo-model.lisai.zip",
                        }
                    }
                ),
            },
        }
    )

    assert catalog.models["direct-model"].download.url == "https://example.org/direct.lisai.zip"
    assert catalog.models["zenodo-model"].download.source.record_id == "12345678"


def test_catalog_download_requires_exactly_one_location():
    with pytest.raises(ValidationError, match="exactly one"):
        DownloadableModelCatalog.model_validate(
            {
                "schema_version": 1,
                "models": {
                    "demo": _entry(
                        {
                            "url": "https://example.org/demo.lisai.zip",
                            "source": {
                                "type": "zenodo",
                                "record_id": "123",
                                "filename": "demo.lisai.zip",
                            },
                        }
                    )
                },
            }
        )


def test_catalog_rejects_invalid_archive_checksum():
    with pytest.raises(ValidationError, match="archive_sha256"):
        DownloadableModelCatalog.model_validate(
            {
                "schema_version": 1,
                "models": {
                    "demo": {
                        **_entry({"url": "https://example.org/demo.lisai.zip"}),
                        "archive_sha256": "not-a-sha256",
                    }
                },
            }
        )


def test_catalog_rejects_unknown_schema_version():
    with pytest.raises(ValidationError, match="Unsupported downloadable-model catalog"):
        DownloadableModelCatalog.model_validate({"schema_version": 2, "models": {}})


def test_catalog_requires_explicit_schema_version():
    with pytest.raises(ValidationError, match="schema_version"):
        DownloadableModelCatalog.model_validate({"models": {}})
