from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Callable, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.promoted_models.sources.zenodo import (
    DEFAULT_ZENODO_TIMEOUT_SECONDS,
    ZenodoSourceError,
    resolve_record_file_download_url,
)

from .catalog import DatasetArchive, DatasetCatalog
from .install import (
    DatasetInstallPlan,
    DatasetInstallResult,
    DatasetInstallSource,
    install_dataset_plan,
)


DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 120.0
DEFAULT_DOWNLOAD_RETRIES = 3
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


class DatasetDownloadError(RuntimeError):
    """Raised when a downloadable dataset archive cannot be fetched."""


class DatasetDownloadIntegrityError(DatasetDownloadError):
    """Raised when a downloaded archive does not match the release catalog."""


@dataclass(frozen=True)
class DatasetDownloadSummary:
    dataset_bytes: int
    noise_model_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.dataset_bytes + self.noise_model_bytes


def _format_bytes(size: int) -> str:
    gib = 1024**3
    mib = 1024**2
    if size >= gib:
        return f"{size / gib:.2f} GiB"
    if size >= mib:
        return f"{size / mib:.1f} MiB"
    return f"{size / 1024:.1f} KiB" if size >= 1024 else f"{size} B"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _progress_line(
    *,
    filename: str,
    current: int,
    total: int,
    stream: TextIO,
) -> None:
    if total > 0:
        percent = min(100.0, (current / total) * 100)
        text = f"\rDownloading {filename}: {percent:5.1f}%  {_format_bytes(current)} / {_format_bytes(total)}"
    else:
        text = f"\rDownloading {filename}: {_format_bytes(current)}"
    print(text, end="", file=stream, flush=True)


def _download_once(
    url: str,
    partial_path: Path,
    *,
    expected_size: int,
    timeout: float,
    stream: TextIO,
) -> None:
    existing = partial_path.stat().st_size if partial_path.is_file() else 0
    headers = {"User-Agent": "LISAI"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", None) or response.getcode()
        resumed = existing > 0 and status == 206
        if existing > 0 and not resumed:
            existing = 0

        mode = "ab" if resumed else "wb"
        current = existing
        _progress_line(
            filename=partial_path.name.removesuffix(".part"),
            current=current,
            total=expected_size,
            stream=stream,
        )
        with partial_path.open(mode) as output:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                current += len(chunk)
                _progress_line(
                    filename=partial_path.name.removesuffix(".part"),
                    current=current,
                    total=expected_size,
                    stream=stream,
                )
    print("", file=stream)


def _download_archive(
    archive: DatasetArchive,
    *,
    url: str,
    downloads_dir: Path,
    timeout: float,
    retries: int,
    stream: TextIO,
) -> Path:
    downloads_dir.mkdir(parents=True, exist_ok=True)
    destination = downloads_dir / archive.filename
    partial = destination.with_name(destination.name + ".part")

    if destination.is_file():
        actual = _sha256_file(destination)
        if actual == archive.sha256:
            print(f"Using verified existing archive: {destination.name}", file=stream)
            return destination
        destination.unlink()

    if partial.exists() and not partial.is_file():
        raise DatasetDownloadError(f"Partial download path is not a file: {partial}")
    if partial.is_file():
        partial_size = partial.stat().st_size
        if partial_size > archive.size_bytes:
            partial.unlink()
        elif partial_size == archive.size_bytes:
            actual_sha256 = _sha256_file(partial)
            if actual_sha256 == archive.sha256:
                os.replace(partial, destination)
                print(f"Using completed partial archive: {destination.name}", file=stream)
                return destination
            partial.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_once(
                url,
                partial,
                expected_size=archive.size_bytes,
                timeout=timeout,
                stream=stream,
            )
            actual_size = partial.stat().st_size
            if actual_size != archive.size_bytes:
                raise DatasetDownloadError(
                    f"Incomplete download for {archive.filename!r}: expected "
                    f"{archive.size_bytes} bytes, got {actual_size}."
                )
            print(f"Verifying SHA256: {archive.filename}", file=stream)
            actual_sha256 = _sha256_file(partial)
            if actual_sha256 != archive.sha256:
                partial.unlink(missing_ok=True)
                raise DatasetDownloadIntegrityError(
                    f"Downloaded archive {archive.filename!r} failed SHA256 verification: "
                    f"expected {archive.sha256}, got {actual_sha256}."
                )
            os.replace(partial, destination)
            return destination
        except DatasetDownloadIntegrityError:
            raise
        except (HTTPError, URLError, HTTPException, OSError, DatasetDownloadError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"Download interrupted ({exc}). Retrying {attempt + 1}/{retries}...",
                file=stream,
            )
            time.sleep(2)

    raise DatasetDownloadError(
        f"Could not download {archive.filename!r} after {retries} attempts: {last_error}"
    ) from last_error


def download_summary(catalog: DatasetCatalog, plan: DatasetInstallPlan) -> DatasetDownloadSummary:
    dataset_bytes = sum(entry.size_bytes for entry in plan.entries if entry.status == "install")
    noise_model_bytes = 0
    if plan.required_noise_models:
        if catalog.shared is None or catalog.shared.noise_models is None:
            raise DatasetDownloadError(
                "Selected datasets require noise models, but the catalog has no noise_models archive."
            )
        noise_model_bytes = catalog.shared.noise_models.archive.size_bytes
    return DatasetDownloadSummary(
        dataset_bytes=dataset_bytes,
        noise_model_bytes=noise_model_bytes,
    )


def _resolve_archive_url(
    record_id: str,
    filename: str,
    *,
    timeout: float,
) -> str:
    try:
        return resolve_record_file_download_url(
            record_id,
            filename,
            timeout=min(timeout, DEFAULT_ZENODO_TIMEOUT_SECONDS),
        )
    except ZenodoSourceError as exc:
        raise DatasetDownloadError(str(exc)) from exc


def download_and_install_dataset_plan(
    catalog: DatasetCatalog,
    plan: DatasetInstallPlan,
    *,
    record_id: str,
    paths: Paths | None = None,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
    stream: TextIO | None = None,
    url_resolver: Callable[[str], str] | None = None,
) -> DatasetInstallResult:
    """Fetch all missing release archives, then pass them to the local installer."""
    resolved_paths = paths or Paths(settings)
    output = sys.stdout if stream is None else stream
    downloads_dir = resolved_paths.datasets_root() / "_downloads"
    local_archives: dict[str, Path] = {}
    noise_archive_path: Path | None = None
    downloaded_paths: list[Path] = []

    def resolve(filename: str) -> str:
        if url_resolver is not None:
            return url_resolver(filename)
        return _resolve_archive_url(record_id, filename, timeout=timeout)

    if plan.required_noise_models:
        assert catalog.shared is not None and catalog.shared.noise_models is not None
        archive = catalog.shared.noise_models.archive
        print(f"\nShared dependency: {archive.filename}", file=output)
        noise_archive_path = _download_archive(
            archive,
            url=resolve(archive.filename),
            downloads_dir=downloads_dir,
            timeout=timeout,
            retries=retries,
            stream=output,
        )
        downloaded_paths.append(noise_archive_path)

    for entry in plan.entries:
        if entry.status == "installed":
            continue
        print(f"\nDownloading dataset: {entry.name}", file=output)
        for archive in entry.dataset.archives:
            archive_path = _download_archive(
                archive,
                url=resolve(archive.filename),
                downloads_dir=downloads_dir,
                timeout=timeout,
                retries=retries,
                stream=output,
            )
            local_archives[archive.filename] = archive_path
            downloaded_paths.append(archive_path)

    source = DatasetInstallSource(
        archives=local_archives,
        noise_models_archive=noise_archive_path,
        verified_archives=frozenset(local_archives),
        noise_models_verified=noise_archive_path is not None,
    )
    result = install_dataset_plan(
        catalog,
        plan,
        source=source,
        paths=resolved_paths,
        stream=output,
    )

    # Only remove complete downloads after the entire installation succeeds. A
    # failed install leaves them in _downloads so the user does not have to fetch
    # tens of gigabytes again.
    for path in downloaded_paths:
        path.unlink(missing_ok=True)
    return result


__all__ = [
    "DEFAULT_DOWNLOAD_RETRIES",
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
    "DatasetDownloadError",
    "DatasetDownloadIntegrityError",
    "DatasetDownloadSummary",
    "download_and_install_dataset_plan",
    "download_summary",
]
