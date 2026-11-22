from __future__ import annotations

import os
from dataclasses import dataclass
from time import monotonic
from typing import Callable
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


@dataclass(frozen=True)
class InferenceProgressEvent:
    """Completed inference work, independent of tqdm, Qt and file saving.

    Frame, tile and sample indices are 1-based *completed* counts when present.
    ``total=None`` denotes an indeterminate setup/finalization stage.
    """

    stage: str
    completed: int = 0
    total: int | None = None
    frame_index: int | None = None
    frame_total: int | None = None
    tile_index: int | None = None
    tile_total: int | None = None
    sample_index: int | None = None
    sample_total: int | None = None


class InferenceProgressReporter:
    """Account for real completed forward passes, throttling display callbacks.

    Cancellation checks remain unthrottled and run at safe inference boundaries.
    No callback means near-zero cost and preserves current prediction behavior.
    """

    def __init__(
        self, on_progress: Callable[[InferenceProgressEvent], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        *, throttle_seconds: float = 0.05,
    ):
        self.on_progress = on_progress
        self.cancel_check = cancel_check
        self.throttle_seconds = float(throttle_seconds)
        self.completed = 0
        self.total = 0
        self.frame_total = 0
        self.sample_total = 1
        self._last_emission = float('-inf')

    def checkpoint(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    def stage(self, stage: str) -> None:
        self.checkpoint()
        if self.on_progress is not None:
            total = self.total if stage == 'done' else None
            self.on_progress(InferenceProgressEvent(stage, self.completed, total))

    def begin(self, *, total: int, frames: int, samples: int) -> None:
        self.checkpoint()
        self.total = int(total)
        self.frame_total = int(frames)
        self.sample_total = int(samples)
        self.completed = 0
        self._last_emission = monotonic()
        if self.on_progress is not None:
            self.on_progress(InferenceProgressEvent('inference', 0, self.total))

    def advance(
        self, count: int, *, frame_index: int, tile_index: int,
        tile_total: int, sample_index: int,
    ) -> None:
        self.checkpoint()
        self.completed += int(count)
        if self.completed > self.total:
            raise RuntimeError('Inference progress exceeded planned work.')
        if self.on_progress is None:
            return
        now = monotonic()
        if self.completed != self.total and now - self._last_emission < self.throttle_seconds:
            return
        self._last_emission = now
        self.on_progress(InferenceProgressEvent(
            'inference', self.completed, self.total,
            frame_index=frame_index, frame_total=self.frame_total,
            tile_index=tile_index, tile_total=tile_total,
            sample_index=sample_index, sample_total=self.sample_total,
        ))


def count_tiles(image_shape, tiling_size: int | None) -> int:
    """Match the engine's odd-pixel crop and find_best_tile geometry.

    This is arithmetic only: never allocate a dummy full-sized tensor just to
    estimate work. The engine uses the same tile search when splitting inputs.
    """
    from lisai.data.utils.tiling import find_best_tile

    h, w = (int(v) for v in image_shape[-2:])
    h -= h % 2
    w -= w % 2
    if tiling_size is None or min(h, w) < tiling_size:
        return 1
    size = int(tiling_size) - int(tiling_size) % 2
    tile_h, _ = find_best_tile(h, 100, size)
    tile_w, _ = find_best_tile(w, 100, size)
    return ((h + tile_h - 1) // tile_h) * ((w + tile_w - 1) // tile_w)
