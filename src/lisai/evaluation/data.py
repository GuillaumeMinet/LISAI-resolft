"""Evaluation data reconstruction helpers.

This module owns the evaluation-side data preparation logic that depends on a
`SavedTrainingRun`: resolving the dataset location, applying evaluation-only
overrides, and building a sample source for `run_evaluate`.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from lisai.config import settings
from lisai.config.io import deep_merge
from lisai.config.models.training import DataSection
from lisai.data.data_loaders.dataset_io import load_image
from lisai.data.data_loaders.split_manifest import manifest_split_entries
from lisai.data.data_loaders.transforms import apply_additional_transforms, apply_inp_transformations
from lisai.data.dataset_registry import load_dataset_info
from lisai.data.utils import crop_center, make_pair_4d
from lisai.infra.paths import Paths
from lisai.lib.upsamp.artificial_movement import apply_movement

from .saved_run import SavedTrainingRun


@dataclass(frozen=True)
class EvalSample:
    """Tensor-ready input/GT pair for one model call."""

    x: torch.Tensor
    y: torch.Tensor | None

    @classmethod
    def from_numpy(cls, x_np: np.ndarray, y_np: np.ndarray | None) -> "EvalSample":
        """Convert one numpy input/GT pair into float32 tensors."""
        return cls(
            x=torch.from_numpy(np.asarray(x_np)).to(torch.float32),
            y=torch.from_numpy(np.asarray(y_np)).to(torch.float32) if y_np is not None else None,
        )


@dataclass(frozen=True)
class EvalItem:
    """File-backed evaluation unit that owns sample selection and naming."""

    name: str
    inp_path: Path
    gt_path: Path | None
    split: str
    file_index: int
    data_format: str
    sample_count: int
    time_indices: tuple[int | None, ...]

    def __len__(self) -> int:
        """Return the number of model calls represented by this item."""
        return self.sample_count

    def iter_samples(self, config: DataSection) -> Iterator[tuple[int, EvalSample]]:
        """Yield `(sample_index, EvalSample)` pairs prepared from this item."""
        inp_img, gt_img = _prepare_eval_item_arrays(self, config=config)
        if inp_img is None:
            return

        for sample_index in self.iter_sample_indices():
            yield sample_index, self.sample_from_arrays(inp_img, gt_img, sample_index)

    def iter_sample_indices(self) -> range:
        """Return the valid sample indices for this item."""
        return range(self.sample_count)

    def sample_from_arrays(
        self,
        inp_img: np.ndarray,
        gt_img: np.ndarray | None,
        sample_index: int,
    ) -> EvalSample:
        """Select one sample from prepared arrays and convert it to tensors."""
        self.validate_sample_index(inp_img, gt_img, sample_index)
        x_np = inp_img[sample_index]
        y_np = gt_img[sample_index] if gt_img is not None else None
        return EvalSample.from_numpy(x_np, y_np)

    def validate_sample_index(
        self,
        inp_img: np.ndarray,
        gt_img: np.ndarray | None,
        sample_index: int,
    ) -> None:
        """Raise if `sample_index` cannot be selected from input or GT arrays."""
        if sample_index >= inp_img.shape[0]:
            raise IndexError(f"Sample index {sample_index} out of range for {self.inp_path}.")
        if gt_img is not None and sample_index >= gt_img.shape[0]:
            raise IndexError(f"Sample index {sample_index} out of range for {self.gt_path}.")

    def sample_time_index(self, sample_index: int) -> int | None:
        """Return the original timelapse index for a sample when available."""
        return self.time_indices[sample_index]

    def sample_name(self, sample_index: int) -> str:
        """Return the output/metrics name for one selected sample."""
        if self.sample_count == 1:
            return self.name

        time_index = self.sample_time_index(sample_index)
        suffix = sample_index if time_index is None else time_index
        return f"{self.name}_{suffix}"

    def sample_sort_key(self, sample_index: int) -> int:
        """Return the ordering key used when regrouping item-level outputs."""
        time_index = self.sample_time_index(sample_index)
        return sample_index if time_index is None else time_index


class EvalSampleSource:
    """Ordered source of evaluation items and their prepared samples."""

    def __init__(
        self,
        *,
        items: Sequence[EvalItem],
        config: DataSection,
        split_manifest: Mapping[str, Any] | None = None,
    ):
        """Store file-level evaluation items and their resolved data config."""
        self.config = config
        self.items = tuple(items)
        self.split_manifest = dict(split_manifest) if split_manifest is not None else None

    @classmethod
    def from_config(
        cls,
        config: DataSection,
        *,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> "EvalSampleSource":
        """Build an item source from a resolved evaluation data config."""
        return cls(
            items=cls.build_items(config, split_manifest=split_manifest),
            config=config,
            split_manifest=split_manifest,
        )

    @staticmethod
    def build_items(
        config: DataSection,
        *,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[EvalItem, ...]:
        """Resolve one file-level evaluation item per input/GT pair."""
        if config.data_dir is None:
            raise ValueError("`data_dir` must be provided for evaluation data loading.")
        if config.input is None and split_manifest is None:
            raise ValueError("`input` must be provided for evaluation data loading.")

        split = getattr(config, "split", "test")
        if split_manifest is not None:
            inp_files, gt_files = _manifest_eval_files(config, split_manifest, split)
        else:
            inp_dir = config.data_dir / config.input / split
            inp_files = _collect_split_files(inp_dir, config.filters)
            if not inp_files:
                raise FileNotFoundError(f"No input files found in {inp_dir} with filters={config.filters}.")

            gt_files: list[Path] | None = None
            if config.target is not None:
                gt_dir = config.data_dir / config.target / split
                gt_files = _collect_split_files(gt_dir, config.filters)
                if len(inp_files) != len(gt_files):
                    raise ValueError(f"Found #{len(inp_files)} inp_files and #{len(gt_files)} gt_files")

        items = []
        for index, inp_path in enumerate(inp_files):
            gt_path = gt_files[index] if gt_files is not None else None
            sample_count = _count_eval_samples(inp_path=inp_path, gt_path=gt_path, config=config)
            if sample_count == 0:
                continue
            data_format = config.resolved_data_format
            items.append(
                EvalItem(
                    name=inp_path.stem,
                    inp_path=inp_path,
                    gt_path=gt_path,
                    split=split,
                    file_index=index,
                    data_format=data_format,
                    sample_count=sample_count,
                    time_indices=_time_indices_for_item(
                        data_format=data_format,
                        sample_count=sample_count,
                        config=config,
                    ),
                )
            )
        return tuple(items)

    def __len__(self) -> int:
        """Return the total number of model calls across all items."""
        return sum(len(item) for item in self.items)

    def iter_items(self) -> Iterator[EvalItem]:
        """Iterate over file-level evaluation items."""
        return iter(self.items)

    def __iter__(self) -> Iterator[EvalSample]:
        """Iterate over prepared samples without item metadata."""
        for item in self.items:
            for _, sample in item.iter_samples(self.config):
                yield sample


def resolve_dataset_info(dataset_name: str | None) -> dict[str, Any] | None:
    """Load dataset-registry metadata for a dataset name when available."""
    if not dataset_name:
        return None

    paths = Paths(settings)
    return load_dataset_info(paths.dataset_registry_path(), dataset_name)


def resolve_eval_data_dir(saved_run: SavedTrainingRun, data_cfg: Mapping[str, Any]) -> Path | None:
    """Resolve the dataset directory used for evaluation from saved config and overrides."""
    for key in ("data_dir", "full_data_path"):
        value = data_cfg.get(key)
        if value:
            return Path(value)

    if data_cfg.get("canonical_load") is False:
        return None

    dataset_name = data_cfg.get("dataset_name") or saved_run.dataset_name
    if not dataset_name:
        return None

    subfolder = data_cfg.get("subfolder")
    if subfolder is None:
        subfolder = saved_run.data_subfolder

    paths = Paths(settings)
    return paths.dataset_dir(dataset_name=dataset_name, data_subfolder=subfolder or "")


def _collect_split_files(data_dir: Path, filters: list[str]) -> list[Path]:
    """Collect split files matching configured suffix filters."""
    files: list[Path] = []
    for image_filter in filters:
        files += [Path(path) for path in sorted(glob.glob(str(data_dir) + f"/*{image_filter}"))]
    return files


def _manifest_eval_files(
    config: DataSection,
    split_manifest: Mapping[str, Any],
    split: str,
) -> tuple[list[Path], list[Path] | None]:
    """Resolve evaluation file paths from a run split manifest."""
    if config.data_dir is None:
        raise ValueError("`data_dir` must be provided for manifest-based evaluation loading.")

    entries = manifest_split_entries(dict(split_manifest), split)
    input_root = config.data_dir / (config.input or "")
    target_root = config.data_dir / config.target if config.target is not None else None

    inp_files: list[Path] = []
    gt_files: list[Path] | None = [] if target_root is not None else None

    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("input"):
            raise ValueError(f"Invalid split manifest entry in split '{split}': {entry!r}")
        input_rel = Path(str(entry["input"]))
        inp_files.append(input_root / input_rel)

        if gt_files is not None:
            target_rel = entry.get("target")
            if target_rel:
                gt_files.append(target_root / str(target_rel))
            else:
                gt_files.append(target_root / input_rel)

    if not inp_files:
        raise FileNotFoundError(f"Split manifest contains no files for split '{split}'.")
    return inp_files, gt_files


def _normalization_flags(config: DataSection) -> tuple[Any, bool, bool]:
    """Resolve legacy normalization flags from a data config."""
    norm_prm = config.norm_prm or {}
    clip = norm_prm.get("clip", False)
    if isinstance(clip, bool) and clip is True:
        clip = 0
    normalize_data = norm_prm.get("normalize_data", False)
    norm_sig_to_obs = norm_prm.get("normSig2Obs", False)
    if config.target is None and norm_sig_to_obs:
        norm_sig_to_obs = False
    return clip, normalize_data, norm_sig_to_obs


def _prepare_eval_item_arrays(item: EvalItem, *, config: DataSection):
    """Load one EvalItem and apply eval-time preprocessing.
    Returns sample-first input/GT arrays for per-sample tensor conversion."""
    data_format = config.resolved_data_format

    inp_img, gt_img = load_image(
        item.inp_path,
        gt_file=item.gt_path,
        data_format=data_format,
        config=config,
    )
    if inp_img is None:
        return None, None

    inp_img, gt_img = make_pair_4d(inp_img, gt_img)  # [sample idx, channel (context_length), H, W]

    if config.artificial_movement is not None:
        movement_prm = config.artificial_movement.model_dump(exclude_none=True)
        inp_img, gt_img = apply_movement((inp_img, gt_img), movement_prm, volumetric=config.volumetric)

    clip, normalize_data, _ = _normalization_flags(config)
    norm_prm = config.norm_prm or {}

    if not isinstance(clip, bool):
        inp_img[inp_img < clip] = clip
        if gt_img is not None:
            gt_img[gt_img < clip] = clip

    if gt_img is not None and gt_img.shape[-2:] != inp_img.shape[-2:]:
        if gt_img.shape[-1] % inp_img.shape[-1] != 0:
            raise ValueError("Ground-truth width must be an integer multiple of input width.")
        if gt_img.shape[-1] // inp_img.shape[-1] != gt_img.shape[-2] // inp_img.shape[-2]:
            raise ValueError("Ground-truth and input spatial scale factors do not match.")
        downsamp_factor = gt_img.shape[-1] // inp_img.shape[-1]
    else:
        downsamp_factor = 1

    initial_crop = config.initial_crop
    if initial_crop is not None:
        if isinstance(initial_crop, int):
            crop_size = initial_crop // downsamp_factor
        else:
            crop_size = (initial_crop[0] // downsamp_factor, initial_crop[1] // downsamp_factor)
        inp_img = crop_center(inp_img, crop_size)
        if gt_img is not None:
            gt_img = crop_center(gt_img, initial_crop)

    if normalize_data:
        inp_img = (inp_img - norm_prm.get("avgObs")) / norm_prm.get("stdObs")
        if gt_img is not None:
            gt_img = (gt_img - norm_prm.get("avgSig")) / norm_prm.get("stdSig")

    list_datasets = [(inp_img, gt_img)]

    if config.downsampling is not None:
        list_datasets, _ = apply_inp_transformations(
            list_datasets,
            config=config,
            for_training=False,
        )

    apply_additional_transforms(list_datasets, config.inp_transform, config.gt_transform)

    model_norm_prm = config.model_norm_prm
    if model_norm_prm is not None:
        inp_img, gt_img = list_datasets[0]
        inp_img = (inp_img - model_norm_prm.get("data_mean")) / model_norm_prm.get("data_std")
        if gt_img is not None:
            gt_mean = model_norm_prm.get("data_mean_gt")
            gt_std = model_norm_prm.get("data_std_gt")
            gt_img = (gt_img - gt_mean) / gt_std
        list_datasets[0] = (inp_img, gt_img)

    return list_datasets[0]


def _count_eval_samples(*, inp_path: Path, gt_path: Path | None, config: DataSection) -> int:
    """Return how many model calls one input/GT file pair expands to."""
    inp_img, gt_img = load_image(
        inp_path,
        gt_file=gt_path,
        data_format=config.resolved_data_format,
        config=config,
    )
    if inp_img is None:
        return 0

    inp_img, _ = make_pair_4d(inp_img, gt_img)
    return inp_img.shape[0]


def _time_indices_for_item(*, data_format: str, sample_count: int, config: DataSection) -> tuple[int | None, ...]:
    """Map prepared sample indices back to original timelapse indices."""
    if data_format != "timelapse":
        return (None,) * sample_count

    timelapse_prm = config.timelapse_prm
    if timelapse_prm is None or timelapse_prm.context_length is None:
        return tuple(range(sample_count))

    side_frames = timelapse_prm.context_length // 2
    return tuple(range(side_frames, side_frames + sample_count))


def build_eval_source(
    saved_run: SavedTrainingRun,
    *,
    split: str = "test",
    crop_size: int | tuple[int, int] | None = None,
    eval_gt=None,
    data_prm_update: Mapping[str, Any] | None = None,
):
    """Build the evaluation sample source for a saved run and eval overrides."""

    # load data preparation config from the trained model
    data_cfg = dict(saved_run.data_cfg)
    model_norm_prm = dict(saved_run.model_norm_prm) if saved_run.model_norm_prm is not None else None

    # Update parameters from evaluation overrides.
    if eval_gt is not None and data_cfg.get("paired") is False:
        data_cfg["paired"] = True
        data_cfg["target"] = eval_gt
        if model_norm_prm is None:
            model_norm_prm = {}
        model_norm_prm["data_mean_gt"] = 0
        model_norm_prm["data_std_gt"] = 1

    if crop_size is not None:
        data_cfg["initial_crop"] = crop_size

    if data_prm_update is not None:
        data_cfg = deep_merge(data_cfg, dict(data_prm_update))

    data_dir = resolve_eval_data_dir(saved_run, data_cfg)
    if data_dir is None:
        raise ValueError(
            "Could not resolve `data_dir` for evaluation. "
            "Provide it through `data_prm_update={\'data_dir\': \'...path...\'}`."
        )

    dataset_info = resolve_dataset_info(data_cfg.get("dataset_name") or saved_run.dataset_name)

    # build data prep config with updated parameters
    prep_cfg = DataSection.model_validate(data_cfg).resolved(
        data_dir=Path(data_dir),
        norm_prm=saved_run.data_norm_prm,
        dataset_info=dataset_info,
        model_norm_prm=model_norm_prm,
        split=split,
    )

    return EvalSampleSource.from_config(prep_cfg, split_manifest=saved_run.split_manifest)



__all__ = [
    "EvalItem",
    "EvalSample",
    "EvalSampleSource",
    "build_eval_source",
    "resolve_dataset_info",
    "resolve_eval_data_dir",
]
