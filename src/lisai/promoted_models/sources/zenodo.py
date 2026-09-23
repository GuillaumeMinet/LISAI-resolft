from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lisai.promoted_models.catalog_schema import ZenodoDownloadSource


ZENODO_RECORD_API_URL = "https://zenodo.org/api/records/{record_id}"
DEFAULT_ZENODO_TIMEOUT_SECONDS = 30.0


class ZenodoSourceError(RuntimeError):
    """Raised when a Zenodo record or file cannot be resolved."""


def _fetch_record(record_id: str, *, timeout: float) -> dict:
    url = ZENODO_RECORD_API_URL.format(record_id=record_id)
    request = Request(url, headers={"User-Agent": "LISAI"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (HTTPError, URLError, OSError) as exc:
        raise ZenodoSourceError(f"Could not retrieve Zenodo record {record_id!r}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ZenodoSourceError(f"Zenodo record {record_id!r} returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ZenodoSourceError(f"Zenodo record {record_id!r} returned an unexpected response.")
    return payload


def _iter_files(record: dict):
    files = record.get("files", [])
    if isinstance(files, list):
        for item in files:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(files, dict):
        entries = files.get("entries", files)
        if isinstance(entries, dict):
            for key, value in entries.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("key", key)
                    yield item


def _download_url(file_info: dict) -> str | None:
    links = file_info.get("links")
    if not isinstance(links, dict):
        return None
    for key in ("content", "download", "self"):
        value = links.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_record_file_download_url(
    record_id: str,
    filename: str,
    *,
    timeout: float = DEFAULT_ZENODO_TIMEOUT_SECONDS,
) -> str:
    """Resolve one exact file in a Zenodo record to its download URL."""
    record = _fetch_record(record_id, timeout=timeout)
    for file_info in _iter_files(record):
        key = file_info.get("key") or file_info.get("filename")
        if key != filename:
            continue
        url = _download_url(file_info)
        if url is None:
            raise ZenodoSourceError(
                f"Zenodo file {filename!r} in record {record_id!r} "
                "does not expose a download URL."
            )
        return url

    raise ZenodoSourceError(
        f"Zenodo record {record_id!r} does not contain file {filename!r}."
    )


def resolve_download_url(
    source: ZenodoDownloadSource,
    *,
    timeout: float = DEFAULT_ZENODO_TIMEOUT_SECONDS,
) -> str:
    """Resolve a promoted-model Zenodo source to its exact file download URL."""
    return resolve_record_file_download_url(
        source.record_id,
        source.filename,
        timeout=timeout,
    )


__all__ = [
    "DEFAULT_ZENODO_TIMEOUT_SECONDS",
    "ZENODO_RECORD_API_URL",
    "ZenodoSourceError",
    "resolve_download_url",
    "resolve_record_file_download_url",
]
