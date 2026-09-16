from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from pydantic import ValidationError

from lisai.config.models.training import TaskName

from .catalog_schema import (
    DownloadableModelCatalog,
    DownloadableModelCatalogEntry,
    DownloadableModelDownload,
)


DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/GuillaumeMinet/LISAI-resolft/main/downloadable_models.yaml"
)
DEFAULT_CATALOG_TIMEOUT_SECONDS = 30.0


class CatalogUnavailableError(RuntimeError):
    """Raised when the remote downloadable-model catalog cannot be retrieved."""


@dataclass(frozen=True)
class CatalogModel:
    name: str
    task: TaskName
    description: str
    archive_sha256: str
    download: DownloadableModelDownload


def _fetch_catalog_bytes(url: str, *, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "LISAI"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (HTTPError, URLError, OSError) as exc:
        raise CatalogUnavailableError(
            f"Could not retrieve the downloadable-model catalog from {url}: {exc}"
        ) from exc


def load_catalog(
    *,
    catalog_url: str | None = None,
    timeout: float = DEFAULT_CATALOG_TIMEOUT_SECONDS,
) -> DownloadableModelCatalog:
    """Fetch and validate the official downloadable-model catalog."""
    url = catalog_url or DEFAULT_CATALOG_URL
    raw = _fetch_catalog_bytes(url, timeout=timeout)
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid downloadable-model catalog at {url}: {exc}") from exc
    if payload is None:
        payload = {}
    try:
        return DownloadableModelCatalog.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid downloadable-model catalog at {url}: {exc}") from exc


def _public_model(name: str, entry: DownloadableModelCatalogEntry) -> CatalogModel:
    return CatalogModel(
        name=name,
        task=entry.task,
        description=entry.description,
        archive_sha256=entry.archive_sha256,
        download=entry.download,
    )


def list_models(
    *,
    catalog_url: str | None = None,
    timeout: float = DEFAULT_CATALOG_TIMEOUT_SECONDS,
) -> list[CatalogModel]:
    catalog = load_catalog(catalog_url=catalog_url, timeout=timeout)
    return [_public_model(name, catalog.models[name]) for name in sorted(catalog.models)]


def get_model(
    name: str,
    *,
    catalog_url: str | None = None,
    timeout: float = DEFAULT_CATALOG_TIMEOUT_SECONDS,
) -> CatalogModel:
    catalog = load_catalog(catalog_url=catalog_url, timeout=timeout)
    entry = catalog.models.get(name)
    if entry is None:
        raise KeyError(
            f"Unknown downloadable model {name!r}. "
            "Use 'lisai models catalog' to inspect available downloads."
        )
    return _public_model(name, entry)


__all__ = [
    "DEFAULT_CATALOG_TIMEOUT_SECONDS",
    "DEFAULT_CATALOG_URL",
    "CatalogModel",
    "CatalogUnavailableError",
    "get_model",
    "list_models",
    "load_catalog",
]
