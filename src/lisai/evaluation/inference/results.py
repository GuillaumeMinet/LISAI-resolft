"""Small public prediction results; no model/runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lisai.hdn_performance import InferencePlan


@dataclass(frozen=True)
class GPUMemoryInfo:
    """CUDA allocator measurements for one prediction, after model loading.

    Values are process-wide PyTorch-managed memory, not total GPU consumption.
    Multiple concurrent GPU jobs can affect the measured peaks. All values
    are bytes; 'additional' is relative to the allocated baseline and is not
    a guarantee of future free memory.
    """

    device: str
    allocated_before_bytes: int
    reserved_before_bytes: int
    driver_free_before_bytes: int
    device_total_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocated_after_bytes: int
    reserved_after_bytes: int
    driver_free_after_bytes: int

    @property
    def additional_peak_allocated_bytes(self) -> int:
        return max(0, self.peak_allocated_bytes - self.allocated_before_bytes)

    @property
    def reusable_cache_before_bytes(self) -> int:
        return max(0, self.reserved_before_bytes - self.allocated_before_bytes)


@dataclass(frozen=True)
class PredictionInfo:
    """0-based *original input* timepoint alignment.

    source_frame_indices are in output order. black_output_indices identify
    zero-valued output placeholders by their positions in the output time axis.
    A single YX image has the single virtual time index (0,).
    """

    source_frame_indices: tuple[int, ...]
    black_output_indices: tuple[int, ...] = ()
    inference_plan: InferencePlan | None = None
    # Timing is diagnostic, not prediction identity; preserve historical
    # PredictionInfo comparisons between equivalent NumPy and TIFF runs.
    inference_seconds: float | None = field(default=None, compare=False)
    gpu_memory: GPUMemoryInfo | None = field(default=None, compare=False)


@dataclass(frozen=True)
class PredictionResult:
    """Actual files written by one TIFF prediction invocation."""

    output_dir: Path
    prediction_paths: tuple[Path, ...]
    info_by_path: dict[Path, PredictionInfo]
    inference_plan: InferencePlan | None = None
