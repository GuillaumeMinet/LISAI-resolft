from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _normalize_filters(filters: Sequence[str]) -> set[str]:
    return {f".{ext.lower().lstrip('.')}" for ext in filters}


def list_image_files(folder: Path, filters: Sequence[str]) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {folder}")

    allowed = _normalize_filters(filters)
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed
    )


def ensure_output_dir(base_dir: Path, *, overwrite: bool, max_suffix: int = 10) -> Path:
    base_dir = Path(base_dir)

    if base_dir.exists():
        if overwrite:
            base_dir.mkdir(parents=True, exist_ok=True)
            return base_dir

        for i in range(max_suffix):
            candidate = Path(f"{base_dir}_{i}")
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate

        raise RuntimeError(
            f"Could not find a free output directory for '{base_dir}' "
            f"after trying suffixes _0 to _{max_suffix - 1}."
        )

    base_dir.mkdir(parents=True, exist_ok=False)
    return base_dir


def save_histogram(path: Path, histogram: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, histogram)
    return path


def save_norm_prm(path: Path, norm_prm: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(norm_prm, f, indent=4)
    return path


def save_text(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_gmm_parameters(model: Any, path: Path) -> Path:
    """
    Persist current GaussianMixtureNoiseModel parameters in the canonical
    NPZ layout expected by model loading code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trained_weight = model.weight.detach().cpu().numpy()
    min_signal = model.min_signal.detach().cpu().numpy()
    max_signal = model.max_signal.detach().cpu().numpy()
    min_sigma = np.asarray(model.min_sigma)

    np.savez(
        path,
        trained_weight=trained_weight,
        min_signal=min_signal,
        max_signal=max_signal,
        min_sigma=min_sigma,
    )
    return path


# Fit into a private directory and publish only a complete calibration.
# Sibling staging/target directories keep renames on the same filesystem.


def _select_noise_model_destination(
    base_dir: Path, *, overwrite: bool, numbered: bool, max_suffix: int = 10
) -> Path:
    """Resolve the destination without creating a visible incomplete model.

    The legacy file builder numbers collisions; the public array API does not.
    """
    if not base_dir.exists() and not base_dir.is_symlink():
        return base_dir
    if base_dir.is_symlink():
        raise ValueError(f"Refusing to replace symlinked noise-model directory: {base_dir}")
    if overwrite:
        if not base_dir.is_dir():
            raise NotADirectoryError(f"Noise-model location is not a directory: {base_dir}")
        return base_dir
    if not numbered:
        raise FileExistsError(f"Noise model already exists: {base_dir}")
    for i in range(max_suffix):
        candidate = base_dir.with_name(f"{base_dir.name}_{i}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(
        f"Could not find a free output directory for '{base_dir}' "
        f"after trying suffixes _0 to _{max_suffix - 1}."
    )


def _publish_noise_model(staging_dir: Path, target: Path, *, overwrite: bool) -> None:
    """Publish a complete directory, restoring the old one on rename failure.

    Replacing a populated directory cannot be done by one atomic rename on all
    supported platforms. The old directory is first moved to a unique sibling
    backup; if publishing fails, it is restored. A hard crash between those two
    renames may leave the backup alongside the staging directory.
    """
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            raise ValueError(f"Refusing to replace symlinked noise-model directory: {target}")
        if not target.is_dir():
            raise NotADirectoryError(f"Noise-model location is not a directory: {target}")
        if not overwrite:
            raise FileExistsError(f"Noise model already exists: {target}")

        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        target.rename(backup)
        try:
            staging_dir.rename(target)
        except BaseException:
            try:
                backup.rename(target)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"Could not restore previous noise model at {target}; "
                    f"it is preserved at {backup}"
                ) from rollback_error
            raise
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.warn(
                f"Noise model updated at {target}, but old backup remains at {backup}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        staging_dir.rename(target)


@contextmanager
def staged_noise_model_output(
    base_dir: Path, *, overwrite: bool, numbered: bool = False
) -> Iterator[tuple[Path, Path]]:
    """Yield ``(private_work_directory, final_directory)`` and publish on success.

    Failed training, plotting, or serialization leaves no new visible model and
    never modifies an existing model selected for overwrite. This also retains
    the legacy file builder's numbered naming convention when requested.
    """
    base_dir = Path(base_dir)
    destination = _select_noise_model_destination(
        base_dir, overwrite=overwrite, numbered=numbered
    )
    base_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=base_dir.parent))
    try:
        yield work_dir, destination
        _publish_noise_model(work_dir, destination, overwrite=overwrite)
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
