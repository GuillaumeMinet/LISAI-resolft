from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lisai.promoted_models import download
from lisai.promoted_models.catalog import CatalogModel
from lisai.promoted_models.catalog_schema import DownloadableModelDownload


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def promoted_model_downloads_dir(self) -> Path:
        return self.root / "downloads"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _model(data: bytes) -> CatalogModel:
    return CatalogModel(
        name="demo-model",
        task="denoising_hdn",
        description="Demo model",
        archive_sha256=_sha(data),
        download=DownloadableModelDownload(url="https://example.org/demo.lisai.zip"),
    )


def _install_fake_download(monkeypatch, data: bytes):
    def fake_stream(url: str, destination: Path, *, timeout: float) -> str:
        destination.write_bytes(data)
        return _sha(data)

    monkeypatch.setattr(download, "_stream_download", fake_stream)


def test_download_model_saves_to_configured_download_directory(monkeypatch, tmp_path: Path):
    data = b"portable-model"
    model = _model(data)
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: model)
    _install_fake_download(monkeypatch, data)

    result = download.download_model("demo-model", paths=FakePaths(tmp_path))

    assert result.status == "downloaded"
    assert result.archive_path == tmp_path / "downloads" / "demo-model.lisai.zip"
    assert result.archive_path.read_bytes() == data
    assert result.archive_sha256 == _sha(data)


def test_download_model_reuses_existing_verified_archive(monkeypatch, tmp_path: Path):
    data = b"portable-model"
    model = _model(data)
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: model)
    archive = tmp_path / "downloads" / "demo-model.lisai.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(data)
    monkeypatch.setattr(
        download,
        "_resolve_remote_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network not needed")),
    )

    result = download.download_model("demo-model", paths=FakePaths(tmp_path))

    assert result.status == "reused"
    assert result.archive_path == archive


def test_download_model_requires_overwrite_for_wrong_existing_checksum(monkeypatch, tmp_path: Path):
    data = b"portable-model"
    model = _model(data)
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: model)
    archive = tmp_path / "downloads" / "demo-model.lisai.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"wrong")

    with pytest.raises(download.DownloadConflictError) as error:
        download.download_model("demo-model", paths=FakePaths(tmp_path))

    assert error.value.path == archive
    assert archive.read_bytes() == b"wrong"


def test_download_model_overwrites_wrong_archive_only_after_verified_download(
    monkeypatch,
    tmp_path: Path,
):
    data = b"portable-model"
    model = _model(data)
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: model)
    _install_fake_download(monkeypatch, data)
    archive = tmp_path / "downloads" / "demo-model.lisai.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"wrong")

    result = download.download_model(
        "demo-model",
        overwrite=True,
        paths=FakePaths(tmp_path),
    )

    assert result.status == "overwritten"
    assert archive.read_bytes() == data


def test_download_model_removes_partial_file_when_checksum_is_wrong(monkeypatch, tmp_path: Path):
    data = b"portable-model"
    model = _model(data)
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: model)

    def bad_stream(url: str, destination: Path, *, timeout: float) -> str:
        destination.write_bytes(b"corrupt")
        return _sha(b"corrupt")

    monkeypatch.setattr(download, "_stream_download", bad_stream)

    with pytest.raises(download.DownloadIntegrityError, match="SHA256"):
        download.download_model("demo-model", paths=FakePaths(tmp_path))

    downloads_dir = tmp_path / "downloads"
    assert not (downloads_dir / "demo-model.lisai.zip").exists()
    assert not (downloads_dir / "demo-model.lisai.zip.part").exists()

class _ChunkedResponse:
    def __init__(self, chunks, *, content_length=None):
        self._chunks = iter(chunks)
        self.headers = {"Content-Length": content_length} if content_length is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return next(self._chunks, b"")


def test_stream_reports_bytes_and_content_length(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: _ChunkedResponse(
        [b"ab", b"cde"], content_length="5"
    ))
    events = []
    destination = tmp_path / "model.part"
    digest = download._stream_download(
        "https://example.org/demo", destination, timeout=2,
        progress_callback=lambda received, total: events.append((received, total)),
    )
    assert destination.read_bytes() == b"abcde"
    assert digest == _sha(b"abcde")
    assert events == [(2, 5), (5, 5)]


def test_stream_progress_without_content_length(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: _ChunkedResponse([b"abc"]))
    events = []
    download._stream_download(
        "https://example.org/demo", tmp_path / "model.part", timeout=2,
        progress_callback=lambda received, total: events.append((received, total)),
    )
    assert events == [(3, None)]


def test_cancelled_download_cleans_partial_and_preserves_existing_archive(monkeypatch, tmp_path):
    data = b"correct-model"
    model = _model(data)
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: _ChunkedResponse(
        [b"first", b"second"]
    ))
    existing = tmp_path / "downloads" / "demo-model.lisai.zip"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old-invalid-but-preserved")
    cancel = False
    events = []

    def report(received, total):
        nonlocal cancel
        events.append(received)
        cancel = True

    with pytest.raises(download.DownloadCancelled, match="cancelled"):
        download.download_model(
            "demo-model", paths=FakePaths(tmp_path), overwrite=True,
            progress_callback=report, cancel_callback=lambda: cancel,
        )
    assert events == [5]
    assert existing.read_bytes() == b"old-invalid-but-preserved"
    assert not existing.with_name(existing.name + ".part").exists()


def test_cancelled_before_fetch_has_no_filesystem_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("catalog should not be contacted")
    ))
    with pytest.raises(download.DownloadCancelled):
        download.download_model("demo-model", paths=FakePaths(tmp_path), cancel_callback=lambda: True)
    assert not (tmp_path / "downloads").exists()


def test_callback_failure_cleans_partial(monkeypatch, tmp_path):
    data = b"abcde"
    monkeypatch.setattr(download.catalog, "get_model", lambda *args, **kwargs: _model(data))
    monkeypatch.setattr(download, "urlopen", lambda request, timeout: _ChunkedResponse([data]))

    def broken(*_args):
        raise RuntimeError("UI handler failed")

    with pytest.raises(RuntimeError, match="UI handler"):
        download.download_model(
            "demo-model", paths=FakePaths(tmp_path), progress_callback=broken,
        )
    assert not list((tmp_path / "downloads").glob("*.part"))
