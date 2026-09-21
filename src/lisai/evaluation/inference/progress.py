from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol, TypeVar

try:
    from tqdm import tqdm

    _tqdm_available = True
except Exception:
    tqdm = None
    _tqdm_available = False


_T = TypeVar("_T")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ProgressLike(Protocol):
    def write(self, message: str) -> None:
        ...

    def track(
        self,
        iterable: Iterable[_T],
        *,
        total: int | None = None,
        desc: str | None = None,
        level: int = 0,
        leave: bool | None = None,
    ) -> Iterable[_T]:
        ...


class InferenceProgress:
    """Small tqdm facade shared by inference stack and tile loops."""

    def __init__(self, *, enabled: bool = True, base_position: int = 0):
        self.enabled = (
            bool(enabled)
            and _tqdm_available
            and not _env_truthy("LISAI_DISABLE_TQDM")
        )
        self.base_position = int(base_position)

    def write(self, message: str) -> None:
        if self.enabled and tqdm is not None:
            tqdm.write(message)
            return
        print(message)

    def track(
        self,
        iterable: Iterable[_T],
        *,
        total: int | None = None,
        desc: str | None = None,
        level: int = 0,
        leave: bool | None = None,
    ) -> Iterable[_T]:
        if not self.enabled or tqdm is None:
            return iterable

        if leave is None:
            leave = level <= 0
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            position=self.base_position + int(level),
            leave=leave,
        )


def ensure_progress(
    progress: ProgressLike | None = None,
    *,
    enabled: bool = True,
) -> ProgressLike:
    if progress is not None:
        return progress
    return InferenceProgress(enabled=enabled)
