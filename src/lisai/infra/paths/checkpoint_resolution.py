from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_STATE_DICT_EPOCH_RE = re.compile(r"^model_epoch_(\d+)_state_dict\.pt$")
_FULL_MODEL_EPOCH_RE = re.compile(r"^model_epoch_(\d+)\.pt$")


def _epoch_checkpoint_pattern(load_method: str) -> tuple[str, re.Pattern[str]]:
    if load_method == "state_dict":
        return "model_epoch_*_state_dict.pt", _STATE_DICT_EPOCH_RE
    if load_method == "full_model":
        return "model_epoch_*.pt", _FULL_MODEL_EPOCH_RE
    raise ValueError(f"Unknown load_method: {load_method}")


def _highest_epoch_checkpoint(checkpoints_dir: Path, *, load_method: str) -> Path | None:
    glob_pattern, filename_pattern = _epoch_checkpoint_pattern(load_method)
    best: tuple[int, Path] | None = None

    for checkpoint_path in checkpoints_dir.glob(glob_pattern):
        match = filename_pattern.fullmatch(checkpoint_path.name)
        if match is None:
            continue

        epoch = int(match.group(1))
        if best is None or epoch > best[0]:
            best = (epoch, checkpoint_path)

    return None if best is None else best[1]


def iter_checkpoint_candidates(
    *,
    paths: Any,
    run_dir: str | Path,
    load_methods: Iterable[str],
    best_or_last: str,
    epoch_number: int | None = None,
) -> Iterator[tuple[str, Path]]:
    """Yield checkpoint paths in the order selectors should be resolved."""
    methods = tuple(load_methods)
    run_dir = Path(run_dir)

    if epoch_number is not None:
        for method in methods:
            yield method, paths.checkpoint_path(
                run_dir=run_dir,
                load_method=method,
                epoch_number=epoch_number,
            )
        return

    if best_or_last == "both":
        selectors = ("best", "last")
    elif best_or_last in {"best", "last"}:
        selectors = (best_or_last,)
    else:
        raise ValueError("best_or_last must be 'best', 'last', or 'both'.")

    for selector in selectors:
        if selector == "best":
            canonical_candidates: list[tuple[str, Path]] = []
            for method in methods:
                checkpoint_path = paths.checkpoint_path(
                    run_dir=run_dir,
                    load_method=method,
                    best_or_last="best",
                )
                canonical_candidates.append((method, checkpoint_path))
                yield method, checkpoint_path

            for method, canonical_path in canonical_candidates:
                epoch_path = _highest_epoch_checkpoint(canonical_path.parent, load_method=method)
                if epoch_path is not None:
                    yield method, epoch_path
            continue

        for method in methods:
            yield method, paths.checkpoint_path(
                run_dir=run_dir,
                load_method=method,
                best_or_last=selector,
            )


def resolve_checkpoint_path(
    *,
    paths: Any,
    run_dir: str | Path,
    load_methods: Iterable[str],
    best_or_last: str,
    epoch_number: int | None = None,
    missing_description: str = "model checkpoint",
) -> tuple[str, Path]:
    """Resolve the first existing checkpoint matching a selector.

    For ``best``, canonical ``model_best_*`` checkpoints are preferred. If none
    exists, epoch-specific checkpoints are scanned and the highest epoch is used.
    Under current training behavior those epoch-specific files are only written
    when validation improves.
    """
    checked_paths: list[str] = []
    seen_paths: set[str] = set()

    for method, checkpoint_path in iter_checkpoint_candidates(
        paths=paths,
        run_dir=run_dir,
        load_methods=load_methods,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
    ):
        checkpoint_path = Path(checkpoint_path)
        path_text = str(checkpoint_path)
        if path_text not in seen_paths:
            checked_paths.append(path_text)
            seen_paths.add(path_text)
        if checkpoint_path.exists():
            return method, checkpoint_path

    raise FileNotFoundError(
        f"Could not find {missing_description}. Checked:\n" + "\n".join(checked_paths)
    )


__all__ = [
    "iter_checkpoint_candidates",
    "resolve_checkpoint_path",
]
