# lisai/data/preprocess/pipelines/recon_timelapse_upsamp.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import tifffile

from lisai.config import settings

from ..core import MAIN_OUTPUT_KEY, FolderSource, Item, OutputDecl, OutputSpec, Source
from ..transformations import bleach_correct_simple_ratio, crop_center_stack, remove_first_frame
from .base import BasePipeline, PipelineResult
from .subfolders import resolve_source_subfolders

if TYPE_CHECKING:
    from ..run_preprocess import PreprocessRun


@dataclass
class ReconTimelapseSimpleConfig:
    """
    Preprocess reconstructed timelapse stacks (T,Y,X) from dump/ into preprocess/recon/.

    This pipeline is the refactored version of the legacy `recon_timelapse_upsamp` pipeline.

    Operations (all optional):
      - remove_first_frame
      - bleach_correction (simple ratio)
      - center crop
      - clip negatives to 0
    """

    base_subfolder: str = field(
        default="",
        metadata={
            "description": "Optional common parent folder inside dataset dump/recon.",
        },
    )
    input_subfolder: str = field(
        default="",
        metadata={
            "description": "Optional folder containing the primary timelapse files, relative to base_subfolder.",
        },
    )
    dump_subfolder: str = field(
        default="",
        metadata={
            "description": (
                "Deprecated alias for base_subfolder. Kept for backward compatibility with "
                "existing recon_timelapse_simple configs."
            ),
        },
    )
    combine_subfolders: bool = field(
        default=False,
        metadata={
            "description": "If true, scan the immediate subfolders of the dump root and combine their files into one preprocess stream.",
        },
    )
    crop_size: int | None = field(
        default=None,
        metadata={
            "description": "Optional centered crop size applied to each timelapse frame before saving.",
        },
    )
    clip_neg: bool = field(
        default=False,
        metadata={
            "description": "If true, clamp negative pixel values to zero after preprocessing.",
        },
    )
    remove_first: bool = field(
        default=False,
        metadata={
            "description": "If true, drop the first frame of each timelapse before any other processing.",
        },
    )
    bleach_correction: bool = field(
        default=False,
        metadata={
            "description": "If true, apply simple-ratio bleach correction to each timelapse stack.",
        },
    )

    def __post_init__(self) -> None:
        resolve_source_subfolders(
            pipeline_name="recon_timelapse_simple",
            base_subfolder=self.base_subfolder,
            input_subfolder=self.input_subfolder,
            dump_subfolder=self.dump_subfolder,
            legacy_dump_role="base",
        )


class ReconTimelapseSimplePipeline(BasePipeline[ReconTimelapseSimpleConfig]):
    Config = ReconTimelapseSimpleConfig
    name = "recon_timelapse_simple"
    supported_data_types = {"recon"}
    supported_fmts = {"timelapse"}

    def output_spec(self) -> OutputSpec:
        return OutputSpec(
            outputs=(OutputDecl(key=MAIN_OUTPUT_KEY, axes="TYX", role="inp"),),
            save_at_root=True,
        )

    def build_source(self, *, run: PreprocessRun) -> Source:
        common_dump_root = run.paths.dataset_dump_dir(
            dataset_name=run.dataset_name,
            data_type=run.data_type,
            usage=run.usage,
        )
        base_subfolder, input_subfolder = resolve_source_subfolders(
            pipeline_name=self.name,
            base_subfolder=self.cfg.base_subfolder,
            input_subfolder=self.cfg.input_subfolder,
            dump_subfolder=self.cfg.dump_subfolder,
            legacy_dump_role="base",
        )
        role_root = common_dump_root / base_subfolder if base_subfolder else common_dump_root
        input_root = role_root / input_subfolder if input_subfolder else role_root
        exts = tuple(settings.data_cfg.data_types[run.data_type])
        return FolderSource(root=input_root, exts=exts, combine_subfolders=self.cfg.combine_subfolders)

    def process_item(self, *, item: Item) -> Dict[str, np.ndarray]:
        (p,) = item.paths
        stack = tifffile.imread(p)

        if stack.ndim != 3 or stack.shape[0] <= 1:
            raise ValueError(
                f"Pipeline {self.name} expects 3D stack (T,Y,X) with T>1, "
                f"got shape={getattr(stack, 'shape', None)} for {p}"
            )

        if self.cfg.remove_first:
            stack = remove_first_frame(stack)
        if self.cfg.bleach_correction:
            stack = bleach_correct_simple_ratio(stack)
        if self.cfg.crop_size is not None:
            stack = crop_center_stack(stack, self.cfg.crop_size)
        if self.cfg.clip_neg:
            stack = stack.copy()
            stack[stack < 0] = 0

        return {MAIN_OUTPUT_KEY: stack}

    def template_kwargs(self, *, item: Item, outputs: dict[str, np.ndarray]) -> dict[str, object]:
        # data.yml timelapse template expects n_timepoints.
        stack = outputs[MAIN_OUTPUT_KEY]
        if stack.ndim == 3:
            return {"n_timepoints": int(stack.shape[0])}
        return {}

    def init_stats(self) -> dict[str, Any]:
        return {"n_frames": 0, "timepoints": set()}

    def update_stats(self, *, stats: dict[str, Any], item, outputs: dict[str, Any]) -> dict[str, Any]:
        stack = outputs[MAIN_OUTPUT_KEY]
        stats["n_frames"] += int(stack.shape[0])
        stats["timepoints"].add(int(stack.shape[0]))
        return stats

    def make_result(self, *, n_files: int, stats: dict[str, Any]) -> PipelineResult:
        timepoints = sorted(int(x) for x in stats.get("timepoints", set()))
        return PipelineResult(
            n_files=n_files,
            n_frames=int(stats.get("n_frames", 0)),
            timepoints=timepoints or None,
        )
