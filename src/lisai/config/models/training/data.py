from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lisai.lib.upsamp.inp_generators import _deterministic_mltpl_sampling

DownsamplingMethod = Literal["blur", "multiple", "random", "real", "all"]
MovementDirection = Literal[
    "random",
    "h+",
    "h-",
    "v+",
    "v-",
    "h+v+",
    "h+v-",
    "h-v+",
    "h-v-",
]
MovementType = Literal["translation"]


class TimelapseParams(BaseModel):
    """Options used when loading timelapse datasets."""

    model_config = ConfigDict(extra="forbid")

    timelapse_max_frames: int | None = Field(
        default=None,
        description="Optional maximum number of frames to load from each timelapse stack.",
    )
    context_length: int | None = Field(
        default=None,
        description="Odd temporal window size centered on the current frame. Use null to keep the full sequence.",
    )
    shuffle: bool = Field(
        default=False,
        description="Whether timelapse frames should be shuffled before sampling.",
    )
    sampling_seed: int = Field(
        default=0,
        description=(
            "Base seed used to deterministically sample timelapse frames when shuffling is enabled. "
            "The effective per-file seed is derived from this base seed and a stable timelapse identifier."
        ),
    )

    @field_validator("context_length", mode="before")
    @classmethod
    def _normalize_context_length(cls, value):
        if value == "None":
            return None
        return value

    @field_validator("timelapse_max_frames")
    @classmethod
    def _validate_timelapse_max_frames(cls, value: int | None):
        if value is not None and value <= 0:
            raise ValueError("`timelapse_max_frames` must be > 0.")
        return value

    @field_validator("context_length")
    @classmethod
    def _validate_context_length(cls, value: int | None):
        if value is None:
            return value
        if value <= 0:
            raise ValueError("`context_length` must be > 0.")
        if value % 2 == 0:
            raise ValueError("`context_length` must be odd.")
        return value

    @field_validator("sampling_seed")
    @classmethod
    def _validate_sampling_seed(cls, value: int):
        if value < 0:
            raise ValueError("`sampling_seed` must be >= 0.")
        return value


class SplitRatios(BaseModel):
    """File-level split ratios used for unprepared, unpaired datasets."""

    model_config = ConfigDict(extra="forbid")

    train: float = Field(default=0.6, ge=0, le=1, description="Fraction of files assigned to training.")
    val: float = Field(default=0.2, ge=0, le=1, description="Fraction of files assigned to validation.")
    test: float = Field(default=0.2, ge=0, le=1, description="Fraction of files assigned to testing.")

    @model_validator(mode="after")
    def _validate_sum(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("`data.split_manifest.ratios` must sum to 1.")
        if self.train <= 0:
            raise ValueError("`data.split_manifest.ratios.train` must be > 0.")
        if self.val <= 0:
            raise ValueError("`data.split_manifest.ratios.val` must be > 0.")
        return self


class SplitManifestParams(BaseModel):
    """Options for file-level split manifests used by prep_before=false."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed used when creating a new unprepared-data split manifest.",
    )
    ratios: SplitRatios = Field(
        default_factory=SplitRatios,
        description="Train/validation/test file-level split ratios for unprepared datasets.",
    )
    retrain_policy: Literal["reuse", "new"] = Field(
        default="reuse",
        description="For retrain with prep_before=false, reuse the origin split manifest or create a new one.",
    )


class MultipleSnrParams(BaseModel):
    """Options used when selecting a noise level from multi-SNR datasets."""

    model_config = ConfigDict(extra="forbid")

    snr_idx: int | list[int] | Literal["last", "random"] | None = Field(
        default=None,
        description="Noise level selection strategy: a single index, a list of indices, 'last', 'random', or null.",
    )

    @field_validator("snr_idx")
    @classmethod
    def _validate_snr_idx(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            if value not in {"last", "random"}:
                raise ValueError("`snr_idx` must be an int, a list of ints, 'last', 'random', or null.")
            return value
        if isinstance(value, int):
            if value < 0:
                raise ValueError("`snr_idx` must be >= 0.")
            return value
        if isinstance(value, list):
            if not value:
                raise ValueError("`snr_idx` list must not be empty.")
            for idx in value:
                if not isinstance(idx, int) or idx < 0:
                    raise ValueError("`snr_idx` list values must be integers >= 0.")
            return value
        raise ValueError("`snr_idx` must be an int, a list of ints, 'last', 'random', or null.")


class DownsamplingMultipleParams(BaseModel):
    """Parameters specific to the 'multiple' downsampling method."""

    model_config = ConfigDict(extra="forbid")

    fill_factor: float = Field(
        description="Fraction of pixels kept in each generated sparse observation, in the interval (0, 1].",
    )
    random: bool = Field(
        default=False,
        description="Whether each generated observation should use a random sparse sampling pattern.",
    )

    @field_validator("fill_factor")
    @classmethod
    def _validate_fill_factor(cls, value: float):
        if value <= 0 or value > 1:
            raise ValueError("`fill_factor` must be in the interval (0, 1].")
        return value


class DownsamplingParams(BaseModel):
    """Parameters controlling synthetic low-resolution input generation for upsampling training."""

    model_config = ConfigDict(extra="forbid")

    supervised_training: bool = Field(
        default=True,
        description="Whether the downsampled input is paired with a supervised target during training.",
    )
    downsamp_factor: int = Field(
        description="Spatial downsampling factor applied to the input.",
    )
    downsamp_method: DownsamplingMethod = Field(
        description="Downsampling strategy. Supported values are blur, multiple, random, real, and all.",
    )
    multiple_prm: DownsamplingMultipleParams | None = Field(
        default=None,
        description="Extra parameters used when downsamp_method is 'multiple'.",
    )
    sampling_strategy: list[list[int]] | None = Field(
        default=None,
        description="Explicit real-sampling positions as two rows: y indices then x indices.",
    )

    def multiple_input_channels(self) -> int | None:
        if self.downsamp_method != "multiple" or self.multiple_prm is None:
            return None
        return int(self.downsamp_factor**2 * self.multiple_prm.fill_factor)

    @field_validator("downsamp_factor")
    @classmethod
    def _validate_downsamp_factor(cls, value: int):
        if value < 2:
            raise ValueError("`downsamp_factor` must be >= 2.")
        return value

    @model_validator(mode="after")
    def _validate_method_specific_fields(self):
        if self.downsamp_method == "multiple":
            if self.multiple_prm is None:
                raise ValueError("`multiple_prm` is required when `downsamp_method='multiple'`.")

            n_ch = self.multiple_input_channels()
            if n_ch is None or n_ch < 1:
                raise ValueError(
                    "`data.downsampling.multiple_prm.fill_factor` yields zero input channels with the current `downsamp_factor`."
                )

            if not self.multiple_prm.random:
                supported = _deterministic_mltpl_sampling.get(int(self.downsamp_factor), {})
                if n_ch not in supported:
                    raise ValueError(
                        "Deterministic `multiple` downsampling is not implemented for "
                        f"`downsamp_factor={self.downsamp_factor}` and `n_ch={n_ch}`. "
                        "Use `multiple_prm.random: true` or choose a supported fill factor."
                    )

        if self.sampling_strategy is not None:
            if self.downsamp_method != "real":
                raise ValueError("`sampling_strategy` is only supported when `downsamp_method='real'`.")
            if len(self.sampling_strategy) != 2:
                raise ValueError("`sampling_strategy` must contain exactly two rows: y positions and x positions.")
            row_lengths = {len(row) for row in self.sampling_strategy}
            if not row_lengths or 0 in row_lengths or len(row_lengths) != 1:
                raise ValueError("`sampling_strategy` rows must be non-empty and have the same length.")
            for row in self.sampling_strategy:
                for idx in row:
                    if idx < 0 or idx >= self.downsamp_factor:
                        raise ValueError(
                            "`sampling_strategy` values must be within [0, downsamp_factor)."
                        )
        return self


class ArtificialMovementParams(BaseModel):
    """Parameters controlling synthetic motion applied to timelapse sequences."""

    model_config = ConfigDict(extra="forbid")

    movement_type: MovementType = Field(
        default="translation",
        description="Synthetic motion model applied to the stack. Only 'translation' is currently supported.",
    )
    speed: float = Field(
        description="Base movement speed in pixels per frame.",
    )
    direction: MovementDirection = Field(
        description="Movement direction. Use 'random' to sample directions dynamically.",
    )
    nFrames: int | None = Field(
        default=None,
        description="Optional number of frames to move. Use null to use the full input sequence length.",
    )
    dynamic_direction: bool = Field(
        default=False,
        description="Whether direction can change over time. Requires direction='random'.",
    )
    variable_speed: bool = Field(
        default=False,
        description="Whether movement speed can vary during the sequence.",
    )
    keep_center_fixed: bool = Field(
        default=False,
        description="Whether the effective crop should stay centered while movement is applied.",
    )
    keep_input_size: bool = Field(
        default=False,
        description="Whether the transformed output should preserve the original input size.",
    )

    @field_validator("speed")
    @classmethod
    def _validate_speed(cls, value: float):
        if value <= 0:
            raise ValueError("`speed` must be > 0.")
        return value

    @field_validator("nFrames")
    @classmethod
    def _validate_nframes(cls, value: int | None):
        if value is not None and value <= 0:
            raise ValueError("`nFrames` must be > 0 when provided.")
        return value

    @model_validator(mode="after")
    def _validate_direction_flags(self):
        if self.dynamic_direction and self.direction != "random":
            raise ValueError("`dynamic_direction` requires `direction='random'`.")
        return self


class ExperimentDataSection(BaseModel):
    """User-authored data-loading and preprocessing settings for a training run."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    dataset_name: str = Field(
        default="unknown_dataset",
        description="Dataset registry name used to resolve dataset metadata and canonical paths.",
    )
    canonical_load: bool = Field(
        default=True,
        description="Whether dataset paths should be resolved through the canonical project routing rules.",
    )

    # Core loader setup
    prep_before: bool = Field(
        default=True,
        description="Whether configured preprocessing should be applied before patch extraction or batching.",
    )
    already_split: bool = Field(
        default=True,
        description="Whether the dataset already contains separate train/val/test splits on disk.",
    )
    paired: bool = Field(
        default=False,
        description="Whether each input sample is paired with an explicit target sample.",
    )
    input: Optional[str] = Field(
        default=None,
        description="Relative input subpath or key used to locate model inputs inside the dataset.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Relative target or ground-truth subpath used for supervised training.",
    )

    # Legacy aliases still seen in older configs
    inp: Optional[str] = Field(
        default=None,
        description="Legacy alias for input. Prefer using input in new configs.",
    )
    gt: Optional[str] = Field(
        default=None,
        description="Legacy alias for target. Prefer using target in new configs.",
    )

    # Data format selection and filtering
    data_format: Optional[str] = Field(
        default=None,
        description="Dataset format identifier, for example single, timelapse, or other loader-specific formats.",
    )
    filters: list[str] = Field(
        default_factory=lambda: ["tif", "tiff"],
        description="Filename extensions accepted when scanning dataset folders.",
    )

    # Patching and batching
    batch_size: int = Field(
        default=1,
        description="Mini-batch size used by the dataset loader.",
    )
    patch_size: Optional[int] = Field(
        default=None,
        description="Training patch size. Use null to train on full images or dataset-native crops.",
    )
    val_patch_size: Optional[int] = Field(
        default=None,
        description="Validation patch size. Use null to reuse patch_size or evaluate full images.",
    )
    patch_thresh: Optional[float] = Field(
        default=None,
        description="Optional threshold used by patch selection logic to keep informative patches only.",
    )
    select_on_gt: bool = Field(
        default=False,
        description="Whether patch selection should use the target image instead of the input image.",
    )
    augmentation: bool = Field(
        default=False,
        description="Whether built-in data augmentation should be applied during training.",
    )
    initial_crop: Optional[Any] = Field(
        default=None,
        description="Optional crop applied before patch extraction. Shape and meaning depend on the loader implementation.",
    )
    mltpl_noise: bool = Field(
        default=False,
        description="Whether the dataset contains multiple noise observations per sample.",
    )

    # Additional transforms
    inp_transform: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional transform configuration applied to the input before it is fed to the model.",
    )
    gt_transform: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional transform configuration applied to the target or ground truth.",
    )

    # Format-specific parameters
    timelapse_prm: TimelapseParams | None = Field(
        default=None,
        description="Timelapse-specific loading options, used for temporal datasets.",
    )
    mltpl_snr_prm: MultipleSnrParams | None = Field(
        default=None,
        description="Multi-SNR selection options, used when a dataset contains several noise levels per sample.",
    )
    split_manifest: SplitManifestParams = Field(
        default_factory=SplitManifestParams,
        description="File-level split manifest options used only when prep_before is false.",
    )

    # Synthetic input-generation parameters
    downsampling: DownsamplingParams | None = Field(
        default=None,
        description="Upsampling-training settings for generating downsampled model inputs.",
    )
    artificial_movement: ArtificialMovementParams | None = Field(
        default=None,
        description="Upsampling-training settings for adding synthetic motion to temporal inputs.",
    )

    # Normalization inputs
    norm_prm: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional normalization statistics or settings applied to the input data.",
    )
    model_norm_prm: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional normalization settings specific to the model output space.",
    )
    avgObs_per_noise: Optional[list[float]] = Field(
        default=None,
        description="Optional per-noise-level average observation values used by specialized normalization flows.",
    )
    stdObs_per_noise: Optional[list[float]] = Field(
        default=None,
        description="Optional per-noise-level standard deviations used by specialized normalization flows.",
    )

    @model_validator(mode="after")
    def _normalize_aliases(self):
        if "masking" in (self.model_extra or {}):
            raise ValueError("`masking` is no longer supported. Remove `data.masking` from the config.")
        if self.target is None and self.gt is not None:
            self.target = self.gt
        if self.input is None and self.inp is not None:
            self.input = self.inp
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be > 0.")
        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError("`patch_size` must be > 0.")
        if self.val_patch_size is not None and self.val_patch_size <= 0:
            raise ValueError("`val_patch_size` must be > 0.")
        if self.prep_before is False and self.paired:
            raise ValueError("`data.prep_before=false` is only supported for unpaired datasets.")
        if self.prep_before is True and self.already_split is False:
            raise ValueError("`data.already_split=false` is no longer supported when `data.prep_before=true`.")
        return self

    @property
    def resolved_data_format(self) -> str:
        if self.data_format is not None:
            return self.data_format
        dataset_info = getattr(self, "dataset_info", None)
        if dataset_info is not None and dataset_info.get("data_format") is not None:
            return dataset_info["data_format"]
        warnings.warn("Data format not specified, put to 'single' by default.")
        return "single"

    @property
    def model_patch_size(self) -> int | None:
        patch_size = self.patch_size
        if patch_size is None:
            patch_size = self.val_patch_size
        return int(patch_size) if patch_size is not None else None

    @property
    def downsampling_factor(self) -> int:
        if self.downsampling is not None:
            return int(self.downsampling.downsamp_factor)

        legacy_downsampling = getattr(self, "downsamp_prm", None)
        if isinstance(legacy_downsampling, Mapping):
            factor = legacy_downsampling.get("downsamp_factor")
        else:
            factor = getattr(legacy_downsampling, "downsamp_factor", None)
        if factor is not None:
            return int(factor)
        return 1

    def resolved(
        self,
        *,
        data_dir: Path,
        dataset_info: Optional[Mapping[str, Any]] = None,
        norm_prm: Optional[Mapping[str, Any]] = None,
        model_norm_prm: Optional[Mapping[str, Any]] = None,
        split: Optional[str] = None,
        volumetric: Optional[bool] = None,
    ) -> "DataSection":
        updates: Dict[str, Any] = {"data_dir": Path(data_dir)}
        if dataset_info is not None:
            updates["dataset_info"] = dict(dataset_info)
        if norm_prm is not None:
            updates["norm_prm"] = dict(norm_prm)
        if model_norm_prm is not None:
            updates["model_norm_prm"] = dict(model_norm_prm)
        if split is not None:
            updates["split"] = split
        if volumetric is not None:
            updates["volumetric"] = volumetric

        return DataSection.model_validate(self.model_dump(exclude_none=False) | updates)


class DataSection(ExperimentDataSection):
    """Resolved training data settings enriched with runtime-only path and dataset metadata."""

    volumetric: bool = Field(
        default=False,
        description="Whether the resolved dataset should be treated as volumetric data.",
    )
    data_dir: Optional[Path] = Field(
        default=None,
        exclude=True,
        description="Resolved on-disk dataset directory. This is a runtime-only field and is not intended for YAML editing.",
    )
    dataset_info: Optional[Dict[str, Any]] = Field(
        default=None,
        exclude=True,
        description="Resolved dataset metadata injected at runtime from the dataset registry.",
    )
    registry_checked: bool = Field(
        default=False,
        exclude=True,
        description="Whether runtime config resolution already checked the dataset registry.",
    )
    registry_data_type: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Registry data type used to resolve defaults for this dataset.",
    )
