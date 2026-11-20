from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lisai.infra.paths import Paths

from . import catalog
from .catalog import CatalogModel
from .package import sha256_file
from .sources.zenodo import resolve_download_url as resolve_zenodo_download_url


DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class DownloadConflictError(FileExistsError):
    """Raised when a downloaded archive exists but does not match the catalog."""

    def __init__(self, *, path: Path, expected_sha256: str, actual_sha256: str):
        self.path = path
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        super().__init__(
            f"Downloaded archive already exists but does not match the catalog checksum: {path}"
        )


class DownloadIntegrityError(ValueError):
    """Raised when a newly downloaded archive fails catalog checksum verification."""


class ModelDownloadError(RuntimeError):
    """Raised when a model archive cannot be downloaded."""


class DownloadCancelled(ModelDownloadError):
    """Raised when a caller requests cancellation before download completion."""


DownloadProgressCallback = Callable[[int, int | None], None]
DownloadCancelCallback = Callable[[], bool]


def _check_cancel(callback: DownloadCancelCallback | None) -> None:
    if callback is not None and callback():
        raise DownloadCancelled("Model download cancelled.")


@dataclass(frozen=True)
class ModelDownload:
    name: str
    archive_path: Path
    archive_sha256: str
    status: Literal["downloaded", "reused", "overwritten"]


def _resolve_remote_url(model: CatalogModel, *, timeout: float) -> str:
    if model.download.url is not None:
        return model.download.url
    source = model.download.source
    if source is None:
        raise ValueError(f"Downloadable model {model.name!r} has no download location.")
    if source.type == "zenodo":
        return resolve_zenodo_download_url(source, timeout=timeout)
    raise ValueError(f"Unsupported downloadable-model source type: {source.type!r}")


def _stream_download(
    url: str,
    destination: Path,
    *,
    timeout: float,
    progress_callback: DownloadProgressCallback | None = None,
    cancel_callback: DownloadCancelCallback | None = None,
) -> str:
    """Stream one archive, reporting bytes written and checking for cancellation.

    ``total_bytes`` may be None when HTTP Content-Length is unavailable. This
    function stays synchronous and independent of any GUI/event loop.
    """
    _check_cancel(cancel_callback)
    request = Request(url, headers={"User-Agent": "LISAI"})
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
            headers = getattr(response, "headers", None)
            raw_total = headers.get("Content-Length") if headers is not None else None
            try:
                total_bytes = int(raw_total) if raw_total is not None else None
                if total_bytes is not None and total_bytes <= 0:
                    total_bytes = None
            except (TypeError, ValueError):
                total_bytes = None
            received = 0
            while True:
                _check_cancel(cancel_callback)
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if progress_callback is not None:
                    progress_callback(received, total_bytes)
            _check_cancel(cancel_callback)
    except (HTTPError, URLError, OSError) as exc:
        raise ModelDownloadError(f"Could not download model archive from {url}: {exc}") from exc
    return digest.hexdigest()

def download_model(
    name: str,
    *,
    overwrite: bool = False,
    paths: Paths,
    catalog_url: str | None = None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    progress_callback: DownloadProgressCallback | None = None,
    cancel_callback: DownloadCancelCallback | None = None,
) -> ModelDownload:
    """Download a model synchronously, with optional progress/cancel callbacks.

    Progress reports ``(received_bytes, total_bytes_or_None)``. Cancellation is
    cooperative between HTTP chunks; it does not interrupt an active socket
    read or catalog lookup. A cancelled transfer removes its temporary file.
    """
    _check_cancel(cancel_callback)
    model = catalog.get_model(name, catalog_url=catalog_url, timeout=timeout)
    _check_cancel(cancel_callback)
    resolved_paths = paths
    downloads_dir = resolved_paths.promoted_model_downloads_dir()
    try:
        downloads_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelDownloadError(
            f"Could not create promoted-model downloads directory {downloads_dir}: {exc}"
        ) from exc

    archive_path = downloads_dir / f"{model.name}.lisai.zip"
    if archive_path.exists() and not archive_path.is_file():
        raise ModelDownloadError(f"Download target exists and is not a file: {archive_path}")

    existed_before = archive_path.is_file()
    if existed_before:
        _check_cancel(cancel_callback)
        try:
            actual = sha256_file(archive_path)
        except OSError as exc:
            raise ModelDownloadError(
                f"Could not read existing downloaded archive {archive_path}: {exc}"
            ) from exc
        if actual == model.archive_sha256:
            _check_cancel(cancel_callback)
            return ModelDownload(
                name=model.name,
                archive_path=archive_path,
                archive_sha256=actual,
                status="reused",
            )
        if not overwrite:
            raise DownloadConflictError(
                path=archive_path,
                expected_sha256=model.archive_sha256,
                actual_sha256=actual,
            )

    _check_cancel(cancel_callback)
    remote_url = _resolve_remote_url(model, timeout=timeout)
    _check_cancel(cancel_callback)
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    if partial_path.exists():
        if not partial_path.is_file():
            raise ModelDownloadError(
                f"Partial download path exists and is not a file: {partial_path}"
            )
        partial_path.unlink()

    try:
        # Preserve existing callers/tests that patch the original _stream_download
        # signature: only forward optional callback keywords when provided.
        stream_options = {"timeout": timeout}
        if progress_callback is not None:
            stream_options["progress_callback"] = progress_callback
        if cancel_callback is not None:
            stream_options["cancel_callback"] = cancel_callback
        actual = _stream_download(remote_url, partial_path, **stream_options)
        _check_cancel(cancel_callback)
        if actual != model.archive_sha256:
            raise DownloadIntegrityError(
                f"Downloaded archive for {model.name!r} failed SHA256 verification: "
                f"expected {model.archive_sha256}, got {actual}."
            )
        _check_cancel(cancel_callback)
        os.replace(partial_path, archive_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    return ModelDownload(
        name=model.name,
        archive_path=archive_path,
        archive_sha256=actual,
        status="overwritten" if existed_before else "downloaded",
    )


__all__ = [
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
    "DownloadCancelCallback",
    "DownloadCancelled",
    "DownloadProgressCallback",
    "DownloadConflictError",
    "DownloadIntegrityError",
    "ModelDownload",
    "ModelDownloadError",
    "download_model",
]
