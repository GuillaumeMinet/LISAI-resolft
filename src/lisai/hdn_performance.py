"""Public HDN performance contracts; dependency-light and import-safe."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt

class HDNPerformanceOptions(BaseModel):
    """Per-call/configuration HDN controls; omitted fields merge sparsely."""

    model_config = ConfigDict(extra="forbid")
    precision: Literal["auto", "fp16", "fp32"] = "auto"
    # Experimental: batching is opt-in; legacy "auto" resolves to one draw.
    sample_batch_size: Literal["auto"] | Annotated[StrictInt, Field(gt=0)] = 1


@dataclass(frozen=True)
class InferencePlan:
    """Resolved choices for a single, fixed-shape inference run."""

    device: str
    precision: Literal["fp16", "fp32"]
    reuse_encoder: bool
    requested_samples: int
    sample_batch_size: int
    tiling_size: int | None
    effective_input_shape: tuple[int, int]
    source: Literal["auto", "manual", "legacy"]
    reason: str
    free_gpu_memory_bytes: int | None = None
    reusable_cuda_cache_bytes: int | None = None
    estimated_usable_gpu_memory_bytes: int | None = None


class HDNOutOfMemoryError(RuntimeError):
    """One failed HDN CUDA allocation, with a suggested *manual* adjustment."""

    def __init__(self, requested_plan: InferencePlan, original: BaseException):
        self.requested_plan = requested_plan
        self.original_error = original
        # Experimental batching is the first manual adjustment when enabled.
        # At batch 1 the user must change spatial tiling explicitly; no retry
        # is performed by LISAI itself.
        self.suggested_performance = None
        self.suggested_tiling_size: int | None = None
        if requested_plan.sample_batch_size > 1:
            self.suggested_performance = HDNPerformanceOptions(
                precision=requested_plan.precision, sample_batch_size=1,
            )
            self.suggestion_reason = (
                "Experimental batching increases memory usage. "
                "Consider manually retrying with batch 1."
            )
        elif requested_plan.tiling_size is None:
            self.suggested_tiling_size = 512
            self.suggestion_reason = (
                "Full-image inference exceeded GPU memory. Enable spatial "
                "tiling and try 512 px; reduce further if needed. "
                "Smaller tiles may take longer and introduce border artifacts."
            )
        elif requested_plan.tiling_size > 128:
            current = requested_plan.tiling_size
            self.suggested_tiling_size = (
                512 if current > 512 else 256 if current > 256 else 128
            )
            self.suggestion_reason = (
                f"Consider manually reducing tiling to {self.suggested_tiling_size} px. "
                "Smaller tiles may take longer and introduce border artifacts."
            )
        else:
            self.suggestion_reason = (
                "Already using small tiles. Try freeing GPU memory or reducing "
                "the model input dimensions; no safe tile size can be guaranteed."
            )
        super().__init__(
            "HDN CUDA out of memory: "
            f"{requested_plan.requested_samples} samples, {requested_plan.precision.upper()}, "
            f"batch {requested_plan.sample_batch_size}, "
            f"tiling {requested_plan.tiling_size if requested_plan.tiling_size is not None else 'off'}. "
            f"{self.suggestion_reason} CUDA error: {original}"
        )
