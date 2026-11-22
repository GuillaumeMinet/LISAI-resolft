"""Strict input-frame to output-frame correspondence for GUI comparisons.

No Qt, image loading, or model imports. Index semantics follow PredictionInfo:
``source_frame_indices`` are original input indices in output order and
``black_output_indices`` are positions in the *output* array.
"""
from __future__ import annotations

from dataclasses import dataclass

from lisai.evaluation.inference.results import PredictionInfo


def _indices(value, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of integer indices")
    result = tuple(value)
    if any(type(i) is not int for i in result):
        raise ValueError(f"{name} must contain integers, not booleans or other values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    return result


@dataclass(frozen=True)
class FrameAlignment:
    """Mapping over the raw input axis, with None for missing predictions."""

    source_to_output: tuple[int | None, ...]
    black_output_indices: frozenset[int]

    @classmethod
    def from_info(
        cls, info: PredictionInfo, *, input_frames: int, output_frames: int,
    ) -> "FrameAlignment":
        if not isinstance(info, PredictionInfo):
            raise TypeError("Prediction info must be a LISAI PredictionInfo")
        if type(input_frames) is not int or input_frames < 1:
            raise ValueError("input_frames must be a positive integer")
        if type(output_frames) is not int or output_frames < 1:
            raise ValueError("output_frames must be a positive integer")
        sources = _indices(info.source_frame_indices, name="source_frame_indices")
        blacks = _indices(info.black_output_indices, name="black_output_indices")
        if len(sources) != output_frames:
            raise ValueError(
                "Prediction frame count does not match source_frame_indices: "
                f"{output_frames} versus {len(sources)}"
            )
        if any(i < 0 or i >= input_frames for i in sources):
            raise ValueError("source_frame_indices refers to an input frame out of bounds")
        if any(i < 0 or i >= output_frames for i in blacks):
            raise ValueError("black_output_indices refers to an output frame out of bounds")
        mapping: list[int | None] = [None] * input_frames
        for output_index, source_index in enumerate(sources):
            mapping[source_index] = output_index
        return cls(tuple(mapping), frozenset(blacks))

    def prediction_index(self, input_frame_index: int) -> int | None:
        return self.source_to_output[input_frame_index]

    def is_black_placeholder(self, input_frame_index: int) -> bool:
        output_index = self.prediction_index(input_frame_index)
        return output_index is not None and output_index in self.black_output_indices
