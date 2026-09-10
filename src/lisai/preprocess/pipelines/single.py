# lisai/data/preprocess/pipelines/single.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import tifffile

from lisai.config import settings

from ..core import (
    MAIN_OUTPUT_KEY,
    AuxiliaryFolderMatcher,
    AuxiliarySource,
    AuxiliarySubfolderConfig,
    FolderSource,
    Item,
    OutputDecl,
    OutputSpec,
    Source,
)
from ..core.auxiliary import normalize_auxiliary_subfolders
from ..transformations import crop_center_2d
from .base import BasePipeline, PipelineResult
from .subfolders import resolve_source_subfolders

if TYPE_CHECKING:
    from ..run_preprocess import PreprocessRun


@dataclass
class SingleReconConfig:
    """
    Preprocess reconstructed single images from dump/ into preprocess/recon/.
    """

    base_subfolder: str = field(
        default="",
        metadata={
            "description": (
                "Optional common parent folder inside dataset dump/recon. "
                "Role-specific input and auxiliary folders are resolved beneath it."
            ),
        },
    )
    input_subfolder: str = field(
        default="",
        metadata={
            "description": (
                "Optional folder containing the primary input images, relative to base_subfolder. "
                "For new configs, this name is also used as the processed output folder unless "
                "output_subfolder overrides it."
            ),
        },
    )
    dump_subfolder: str = field(
        default="",
        metadata={
            "description": (
                "Deprecated alias for input_subfolder. Kept for backward compatibility with "
                "existing single_recon configs."
            ),
        },
    )
    output_subfolder: str | None = field(
        default=None,
        metadata={
            "description": (
                "Processed-data destination for the primary input stream. If omitted, new "
                "input_subfolder configs mirror that stream name. Set explicitly to null to "
                "save at preprocess/<data_type>/ root, or provide another relative subfolder "
                "to override the mirrored name."
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
            "description": "Optional centered crop size applied to each 2D image before saving.",
        },
    )
    auxiliary_subfolders: dict[str, AuxiliarySubfolderConfig] = field(
        default_factory=dict,
        metadata={
            "description": (
                "Optional named auxiliary image folders relative to base_subfolder. "
                "Each auxiliary is saved as role 'aux' and must explicitly configure matching. "
                "Files can be matched by exact filename or by acquisition timestamp."
            ),
        },
    )

    def __post_init__(self) -> None:
        resolve_source_subfolders(
            pipeline_name="single_recon",
            base_subfolder=self.base_subfolder,
            input_subfolder=self.input_subfolder,
            dump_subfolder=self.dump_subfolder,
            legacy_dump_role="input",
        )


class SingleReconPipeline(BasePipeline[SingleReconConfig]):
    Config = SingleReconConfig
    name = "single_recon"
    supported_data_types = {"recon"}
    supported_fmts = {"single"}

    def __init__(self, cfg: SingleReconConfig):
        super().__init__(cfg)

    @classmethod
    def parse_cfg(cls, raw: dict | None) -> SingleReconConfig:
        cfg = super().parse_cfg(raw)
        setattr(cfg, "_output_subfolder_explicit", bool(raw is not None and "output_subfolder" in raw))
        return cfg

    def _primary_output_subfolder(self) -> str:
        # A non-None value is always an explicit override, including direct dataclass use.
        if self.cfg.output_subfolder is not None:
            return self.cfg.output_subfolder

        # Through normal config parsing, an explicitly provided YAML null means flatten to root.
        if getattr(self.cfg, "_output_subfolder_explicit", False):
            return ""

        # New input_subfolder semantics preserve the meaningful stream name in preprocess/.
        if self.cfg.input_subfolder:
            return self.cfg.input_subfolder

        # Legacy single_recon dump_subfolder selected the source but historically saved at root.
        return ""

    def _auxiliary_cfg(self) -> dict[str, AuxiliarySubfolderConfig]:
        return normalize_auxiliary_subfolders(
            self.cfg.auxiliary_subfolders,
            reserved_names={"inp", "gt"},
        )

    def output_spec(self) -> OutputSpec:
        outputs = [
            OutputDecl(
                key=MAIN_OUTPUT_KEY,
                axes="YX",
                role="inp",
                path=self._primary_output_subfolder(),
            )
        ]
        outputs.extend(
            OutputDecl(key=name, axes="YX", role="aux")
            for name in self._auxiliary_cfg()
        )
        return OutputSpec(outputs=tuple(outputs))

    def build_source(self, *, run: PreprocessRun) -> Source:
        common_dump_root = run.paths.dataset_dump_dir(
            dataset_name=run.dataset_name,
            data_type=run.data_type,
        )
        base_subfolder, input_subfolder = resolve_source_subfolders(
            pipeline_name=self.name,
            base_subfolder=self.cfg.base_subfolder,
            input_subfolder=self.cfg.input_subfolder,
            dump_subfolder=self.cfg.dump_subfolder,
            legacy_dump_role="input",
        )
        role_root = common_dump_root / base_subfolder if base_subfolder else common_dump_root
        primary_root = role_root / input_subfolder if input_subfolder else role_root
        exts = tuple(settings.data_cfg.data_types[run.data_type])

        aux_cfg = self._auxiliary_cfg()
        if aux_cfg and self.cfg.combine_subfolders:
            raise ValueError(
                "single_recon does not support combine_subfolders=true together with auxiliary_subfolders. "
                "Select the primary stream with input_subfolder instead."
            )

        source: Source = FolderSource(
            root=primary_root,
            exts=exts,
            combine_subfolders=self.cfg.combine_subfolders,
        )
        if not aux_cfg:
            return source

        matcher = AuxiliaryFolderMatcher(
            root=role_root,
            auxiliary_subfolders=aux_cfg,
            exts=exts,
            reserved_names={"inp", "gt"},
        )
        return AuxiliarySource(source=source, matcher=matcher)

    def process_item(self, *, item: Item) -> Dict[str, np.ndarray]:
        (p,) = item.paths
        img = tifffile.imread(p)

        if img.ndim != 2:
            raise ValueError(f"SingleReconPipeline expects 2D images, got {img.ndim}D for {p}")

        if self.cfg.crop_size is not None:
            img = crop_center_2d(img, self.cfg.crop_size)

        outputs: Dict[str, np.ndarray] = {MAIN_OUTPUT_KEY: img}

        for name, aux_path in item.auxiliary_paths.items():
            aux = tifffile.imread(aux_path)
            if aux.ndim != 2:
                raise ValueError(
                    f"SingleReconPipeline auxiliary '{name}' expects a 2D image, "
                    f"got {aux.ndim}D for {aux_path}"
                )
            if self.cfg.crop_size is not None:
                aux = crop_center_2d(aux, self.cfg.crop_size)
            outputs[name] = aux

        return outputs

    def template_kwargs(self, *, item: Item, outputs: dict[str, np.ndarray]) -> dict[str, object]:
        return {}

    def make_result(self, *, n_files: int, stats: dict[str, Any]) -> PipelineResult:
        result = PipelineResult(n_files=n_files)
        return result
