from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np
import tifffile

from lisai.config import settings

from ..core import (
    AuxiliaryFolderMatcher,
    AuxiliarySource,
    AuxiliarySubfolderConfig,
    Item,
    OutputDecl,
    OutputSpec,
    Source,
)
from ..core.auxiliary import normalize_auxiliary_subfolders
from ..transformations import crop_center_2d, register_stack_skimage
from .base import BasePipeline, PipelineResult
from .subfolders import resolve_source_subfolders

if TYPE_CHECKING:
    from ..run_preprocess import PreprocessRun


@dataclass(frozen=True)
class PairedFolderSource(Source):
    """Pair files from two sibling folders by exact filename."""

    root: Path
    input_subfolder: str
    gt_subfolder: str
    exts: tuple[str, ...]

    def _files_by_name(self, folder: Path) -> dict[str, Path]:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Paired dataset folder does not exist: {folder}"
            )

        return {
            path.name: path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in self.exts
        }

    def iter_items(self) -> Iterable[Item]:
        input_dir = self.root / self.input_subfolder
        gt_dir = self.root / self.gt_subfolder

        input_files = self._files_by_name(input_dir)
        gt_files = self._files_by_name(gt_dir)

        missing_gt = sorted(set(input_files) - set(gt_files))
        missing_input = sorted(set(gt_files) - set(input_files))

        if missing_gt or missing_input:
            details: list[str] = []

            if missing_gt:
                details.append(f"missing GT for: {missing_gt}")

            if missing_input:
                details.append(f"missing input for: {missing_input}")

            raise ValueError(
                "Paired folders must contain exactly matching filenames; "
                + "; ".join(details)
            )

        for name in sorted(input_files):
            input_path = input_files[name]
            gt_path = gt_files[name]

            yield Item(
                key=input_path.stem,
                paths=(input_path, gt_path),
                source_name=input_path.name,
                source_relpaths=(
                    input_path.relative_to(self.root).as_posix(),
                    gt_path.relative_to(self.root).as_posix(),
                ),
            )


@dataclass
class PairedSingleReconConfig:
    """Preprocess paired 2D reconstructed images stored in two folders."""

    base_subfolder: str = field(
        default="",
        metadata={
            "description": (
                "Optional common parent folder inside dataset dump/recon. Input, GT, and "
                "auxiliary folders are resolved beneath it."
            ),
        },
    )

    dump_subfolder: str = field(
        default="",
        metadata={
            "description": (
                "Deprecated alias for base_subfolder. Kept for backward compatibility with "
                "existing paired_single_recon configs."
            ),
        },
    )

    input_subfolder: str = field(
        default="low",
        metadata={
            "description": (
                "Folder containing the model input images, relative to base_subfolder. "
                "Its name is preserved as the processed input folder."
            ),
        },
    )

    gt_subfolder: str = field(
        default="high",
        metadata={
            "description": (
                "Folder containing the paired evaluation ground-truth images, relative to "
                "base_subfolder. Its name is preserved as the processed GT output folder."
            ),
        },
    )

    registration: bool = field(
        default=False,
        metadata={
            "description": (
                "If true, register each input image to its paired "
                "ground-truth image before saving."
            ),
        },
    )

    crop_size: int | None = field(
        default=None,
        metadata={
            "description": (
                "Optional centered crop size applied identically to each "
                "paired input and ground-truth image."
            ),
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
            pipeline_name="paired_single_recon",
            base_subfolder=self.base_subfolder,
            dump_subfolder=self.dump_subfolder,
            legacy_dump_role="base",
        )


class PairedSingleReconPipeline(BasePipeline[PairedSingleReconConfig]):
    Config = PairedSingleReconConfig
    name = "paired_single_recon"

    supported_data_types = {"recon"}
    supported_fmts = {"single"}

    def _auxiliary_cfg(self) -> dict[str, AuxiliarySubfolderConfig]:
        return normalize_auxiliary_subfolders(
            self.cfg.auxiliary_subfolders,
            reserved_names={"inp", "gt"},
        )

    def output_spec(self) -> OutputSpec:
        outputs = [
            OutputDecl(
                key="inp",
                axes="YX",
                role="inp",
                path=self.cfg.input_subfolder,
            ),
            OutputDecl(
                key="gt",
                axes="YX",
                role="gt",
                path=self.cfg.gt_subfolder,
            ),
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
        base_subfolder, _ = resolve_source_subfolders(
            pipeline_name=self.name,
            base_subfolder=self.cfg.base_subfolder,
            dump_subfolder=self.cfg.dump_subfolder,
            legacy_dump_role="base",
        )
        dump_root = common_dump_root / base_subfolder if base_subfolder else common_dump_root

        exts = tuple(settings.data_cfg.data_types[run.data_type])

        source: Source = PairedFolderSource(
            root=dump_root,
            input_subfolder=self.cfg.input_subfolder,
            gt_subfolder=self.cfg.gt_subfolder,
            exts=exts,
        )

        aux_cfg = self._auxiliary_cfg()
        if not aux_cfg:
            return source

        matcher = AuxiliaryFolderMatcher(
            root=dump_root,
            auxiliary_subfolders=aux_cfg,
            exts=exts,
            reserved_names={"inp", "gt"},
        )
        return AuxiliarySource(source=source, matcher=matcher)

    def process_item(
        self,
        *,
        item: Item,
    ) -> Dict[str, np.ndarray]:
        input_path, gt_path = item.paths

        inp = tifffile.imread(input_path)
        gt = tifffile.imread(gt_path)

        if inp.ndim != 2:
            raise ValueError(
                "PairedSingleReconPipeline expects 2D input images, "
                f"got shape={inp.shape} for {input_path}"
            )

        if gt.ndim != 2:
            raise ValueError(
                "PairedSingleReconPipeline expects 2D GT images, "
                f"got shape={gt.shape} for {gt_path}"
            )

        if inp.shape != gt.shape:
            raise ValueError(
                "Paired input and GT images must have the same shape "
                "before registration/cropping: "
                f"{input_path.name}={inp.shape}, "
                f"{gt_path.name}={gt.shape}"
            )

        if self.cfg.registration:
            pair = np.stack((inp, gt), axis=0)

            pair = register_stack_skimage(
                pair,
                reference_index=1,
                interpolation_order=1,
            )

            inp, gt = pair[0], pair[1]

        if self.cfg.crop_size is not None:
            inp = crop_center_2d(
                inp,
                self.cfg.crop_size,
            )
            gt = crop_center_2d(
                gt,
                self.cfg.crop_size,
            )

        outputs: Dict[str, np.ndarray] = {
            "inp": inp,
            "gt": gt,
        }

        for name, aux_path in item.auxiliary_paths.items():
            aux = tifffile.imread(aux_path)
            if aux.ndim != 2:
                raise ValueError(
                    f"PairedSingleReconPipeline auxiliary '{name}' expects a 2D image, "
                    f"got shape={aux.shape} for {aux_path}"
                )
            if self.cfg.crop_size is not None:
                aux = crop_center_2d(aux, self.cfg.crop_size)
            outputs[name] = aux

        return outputs

    def template_kwargs(
        self,
        *,
        item: Item,
        outputs: dict[str, np.ndarray],
    ) -> dict[str, object]:
        return {}

    def make_result(
        self,
        *,
        n_files: int,
        stats: dict[str, Any],
    ) -> PipelineResult:
        return PipelineResult(
            n_files=n_files,
        )