"""Small public prediction results; no model/runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionInfo:
    """0-based *original input* timepoint alignment.

    source_frame_indices are in output order. black_output_indices identify
    zero-valued output placeholders by their positions in the output time axis.
    A single YX image has the single virtual time index (0,).
    """

    source_frame_indices: tuple[int, ...]
    black_output_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PredictionResult:
    """Actual files written by one TIFF prediction invocation."""

    output_dir: Path
    prediction_paths: tuple[Path, ...]
    info_by_path: dict[Path, PredictionInfo]
