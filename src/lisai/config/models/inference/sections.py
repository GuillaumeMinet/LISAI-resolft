from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ApplyOutputMode: TypeAlias = Literal["default", "in_place", "folder_inside", "folder_outside"]
SaveInputMode: TypeAlias = Literal["always", "never", "if_not_in_place"]
CheckpointSelector = Literal["best", "last", "both"]
PositiveTilingSize: TypeAlias = Annotated[int, Field(gt=0)]
TilingSizePolicy: TypeAlias = PositiveTilingSize | Literal["auto", "off"] | None

OUTPUT_MODE_DESC = (
    "Output placement for apply predictions: default uses the configured inference "
    "directory, in_place writes directly with the input data, folder_inside creates a "
    "dedicated prediction folder inside the input directory, and folder_outside creates "
    "that folder beside the input directory."
)
APPLY_SAVE_FOLDER_DESC = (
    "Explicit output directory for apply predictions. When omitted, output routing "
    "falls back to the local/project inference defaults."
)
IN_PLACE_DESC = (
    "Whether apply outputs should be written alongside the input data. "
    "Use false to explicitly force normal inference-directory routing."
)
EVALUATE_SAVE_FOLDER_DESC = (
    "Output folder for evaluation artifacts. Use null to save inside the model run directory."
)
EPOCH_NUMBER_DESC = (
    "Explicit checkpoint epoch number to load. Use null to select the checkpoint "
    "through best_or_last instead."
)
BEST_OR_LAST_DESC = "Checkpoint selector used when epoch_number is null. Supported values are 'best', 'last', and 'both'."
FILTERS_DESC = "File extensions accepted when apply input points to a directory."
SKIP_IF_CONTAIN_DESC = "Optional substrings; matching filenames are skipped during apply."
CROP_SIZE_DESC = "Optional center crop used before inference. Use a single integer for a square crop or a height,width tuple."
KEEP_ORIGINAL_SHAPE_DESC = "Whether outputs should be padded back to the original spatial size after cropped inference."
TILING_SIZE_DESC = (
    "Tile size used for patch-wise inference. Use 'auto' or null to fall back "
    "to the saved model default, use a positive integer to force a tile size, "
    "or use 'off' to disable tiling."
)
STACK_SELECTION_IDX_DESC = "Optional stack or channel index selected before converting the input to a 4D tensor."
TIMELAPSE_MAX_DESC = "Optional maximum number of timelapse frames to process."
LVAE_NUM_SAMPLES_DESC = "Number of stochastic samples drawn when running LVAE models."
LVAE_SAVE_SAMPLES_DESC = "Whether individual LVAE samples should be saved in addition to the main prediction."
DENORMALIZE_OUTPUT_DESC = "Whether model outputs should be converted back from normalized model space before saving."
SAVE_INPUT_MODE_DESC = (
    "Policy controlling whether apply also saves its input: always saves it, never omits it, "
    "and if_not_in_place saves it unless predictions are written directly with the source data."
)
DOWNSAMP_DESC = "Optional spatial downsampling factor applied to the apply input before inference."
FILL_FACTOR_DESC = (
    "Optional fill factor used with deterministic 'multiple' apply downsampling. "
    "When set, apply uses generate_downsamp_inp with random=false."
)
APPLY_COLOR_CODE_DESC = "Whether an additional color-coded visualization should be saved for volumetric apply outputs."
DARK_FRAME_CONTEXT_LENGTH_DESC = (
    "Whether missing timelapse context frames should be replaced with dark frames "
    "instead of skipping sequence edges."
)
COLORMAP_DESC = "Matplotlib colormap name used for volumetric color-coding."
SATURATION_DESC = "Contrast boost applied to the color-coded volumetric visualization."
ADD_COLORBAR_DESC = "Whether a colorbar should be appended to the color-coded volumetric visualization."
ZSTEP_DESC = "Physical or logical spacing between z slices used when scaling the volumetric colorbar."
METRICS_LIST_DESC = "Metrics to compute during evaluation, for example ['psnr', 'ssim']. Use null to skip metrics."
OVERWRITE_DESC = "Whether an existing save folder may be overwritten."
EVAL_GT_DESC = (
    "Optional evaluation ground-truth path/key. If omitted, evaluation uses the dataset registry eval_gt "
    "default when present, otherwise the training target. Use @training to force the saved training target "
    "and @none to evaluate without ground truth."
)
DATA_PRM_UPDATE_DESC = "Optional extra data-loader overrides such as {'data_dir': '...'} or {'subfolder': '...'} used during evaluation."
CH_OUT_DESC = (
    "Output channel count forwarded to inference. Defaults to 1 for evaluation "
    "because models are trained to predict a single target frame even when inputs "
    "contain multiple context channels."
)
SPLIT_DESC = "Dataset split to evaluate, typically 'test' or 'val'."
LIMIT_N_IMGS_DESC = "Optional cap on the number of input images or batches processed."


def _normalize_tiling_size_policy(value):
    if value is False:
        return "off"
    if value is True:
        raise ValueError("tiling_size must be a positive integer, 'auto', 'off', or null.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto"}:
            return "auto"
        if normalized in {"off", "none", "disable", "disabled"}:
            return "off"
        return normalized
    return value


class CheckpointDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch_number: int | None = Field(default=None, description=EPOCH_NUMBER_DESC)
    best_or_last: CheckpointSelector = Field(default="best", description=BEST_OR_LAST_DESC)


class CheckpointOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch_number: int | None = Field(default=None, description=EPOCH_NUMBER_DESC)
    best_or_last: CheckpointSelector | None = Field(default=None, description=BEST_OR_LAST_DESC)


class ApplyInputDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: list[str] = Field(default_factory=lambda: ["tiff", "tif"], description=FILTERS_DESC)
    skip_if_contain: list[str] | None = Field(default=None, description=SKIP_IF_CONTAIN_DESC)
    stack_selection_idx: int | None = Field(default=None, description=STACK_SELECTION_IDX_DESC)
    limit_n_imgs: int | None = Field(default=None, gt=0, description=LIMIT_N_IMGS_DESC)
    timelapse_max: int | None = Field(default=None, description=TIMELAPSE_MAX_DESC)


class ApplyInputOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: list[str] | None = Field(default=None, description=FILTERS_DESC)
    skip_if_contain: list[str] | None = Field(default=None, description=SKIP_IF_CONTAIN_DESC)
    stack_selection_idx: int | None = Field(default=None, description=STACK_SELECTION_IDX_DESC)
    limit_n_imgs: int | None = Field(default=None, gt=0, description=LIMIT_N_IMGS_DESC)
    timelapse_max: int | None = Field(default=None, description=TIMELAPSE_MAX_DESC)


class ApplyInferenceDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    keep_original_shape: bool = Field(default=True, description=KEEP_ORIGINAL_SHAPE_DESC)
    tiling_size: TilingSizePolicy = Field(default="auto", description=TILING_SIZE_DESC)
    lvae_num_samples: int | None = Field(default=30, description=LVAE_NUM_SAMPLES_DESC)
    downsamp: int | None = Field(default=None, description=DOWNSAMP_DESC)
    fill_factor: float | None = Field(default=None, gt=0, le=1, description=FILL_FACTOR_DESC)
    dark_frame_context_length: bool = Field(default=False, description=DARK_FRAME_CONTEXT_LENGTH_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class ApplyInferenceOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    keep_original_shape: bool | None = Field(default=None, description=KEEP_ORIGINAL_SHAPE_DESC)
    tiling_size: TilingSizePolicy = Field(default=None, description=TILING_SIZE_DESC)
    lvae_num_samples: int | None = Field(default=None, description=LVAE_NUM_SAMPLES_DESC)
    downsamp: int | None = Field(default=None, description=DOWNSAMP_DESC)
    fill_factor: float | None = Field(default=None, gt=0, le=1, description=FILL_FACTOR_DESC)
    dark_frame_context_length: bool | None = Field(default=None, description=DARK_FRAME_CONTEXT_LENGTH_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class ColorCodeDefaults(BaseModel):
    """Complete color-coding settings used for volumetric apply outputs."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description=APPLY_COLOR_CODE_DESC)
    colormap: str = Field(default="turbo", description=COLORMAP_DESC)
    saturation: float = Field(default=0.35, description=SATURATION_DESC)
    add_colorbar: bool = Field(default=True, description=ADD_COLORBAR_DESC)
    zstep: float = Field(default=0.4, description=ZSTEP_DESC)


class ColorCodeOverrides(BaseModel):
    """Sparse user overrides for volumetric color-coding settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = Field(default=None, description=APPLY_COLOR_CODE_DESC)
    colormap: str | None = Field(default=None, description=COLORMAP_DESC)
    saturation: float | None = Field(default=None, description=SATURATION_DESC)
    add_colorbar: bool | None = Field(default=None, description=ADD_COLORBAR_DESC)
    zstep: float | None = Field(default=None, description=ZSTEP_DESC)


class ApplyPostprocessDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    denormalize: bool = Field(default=True, description=DENORMALIZE_OUTPUT_DESC)
    color_code: ColorCodeDefaults = Field(default_factory=ColorCodeDefaults)


class ApplyPostprocessOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    denormalize: bool | None = Field(default=None, description=DENORMALIZE_OUTPUT_DESC)
    color_code: ColorCodeOverrides | None = None


class ApplySavingDefaults(BaseModel):
    """Saving behavior. Routing values default to null so local_config remains the baseline."""

    model_config = ConfigDict(extra="forbid")

    lvae_save_samples: bool = Field(default=True, description=LVAE_SAVE_SAMPLES_DESC)
    mode: ApplyOutputMode | None = Field(default=None, description=OUTPUT_MODE_DESC)
    save_input_mode: SaveInputMode | None = Field(default=None, description=SAVE_INPUT_MODE_DESC)
    save_folder: str | None = Field(default=None, description=APPLY_SAVE_FOLDER_DESC)
    in_place: bool | None = Field(default=None, description=IN_PLACE_DESC)

    @field_validator("save_folder", mode="before")
    @classmethod
    def _normalize_save_folder(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_exclusive_output_choice(self):
        selected = [self.mode is not None, self.save_folder is not None, self.in_place is not None]
        if sum(selected) > 1:
            raise ValueError(
                "apply.saving.mode, apply.saving.save_folder, and apply.saving.in_place "
                "are mutually exclusive."
            )
        return self


class ApplySavingOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lvae_save_samples: bool | None = Field(default=None, description=LVAE_SAVE_SAMPLES_DESC)
    mode: ApplyOutputMode | None = Field(default=None, description=OUTPUT_MODE_DESC)
    save_input_mode: SaveInputMode | None = Field(default=None, description=SAVE_INPUT_MODE_DESC)
    save_folder: str | None = Field(default=None, description=APPLY_SAVE_FOLDER_DESC)
    in_place: bool | None = Field(default=None, description=IN_PLACE_DESC)

    @field_validator("save_folder", mode="before")
    @classmethod
    def _normalize_save_folder(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_exclusive_output_choice(self):
        selected = [self.mode is not None, self.save_folder is not None, self.in_place is not None]
        if sum(selected) > 1:
            raise ValueError(
                "apply.saving.mode, apply.saving.save_folder, and apply.saving.in_place "
                "are mutually exclusive."
            )
        return self


class ApplyDefaults(BaseModel):
    """Canonical LISAI defaults for the `apply` inference section."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointDefaults = Field(default_factory=CheckpointDefaults)
    input: ApplyInputDefaults = Field(default_factory=ApplyInputDefaults)
    inference: ApplyInferenceDefaults = Field(default_factory=ApplyInferenceDefaults)
    postprocess: ApplyPostprocessDefaults = Field(default_factory=ApplyPostprocessDefaults)
    saving: ApplySavingDefaults = Field(default_factory=ApplySavingDefaults)


_APPLY_LEGACY_GROUPS = {
    "checkpoint": {"epoch_number", "best_or_last"},
    "input": {"filters", "skip_if_contain", "stack_selection_idx", "limit_n_imgs", "timelapse_max"},
    "inference": {
        "crop_size", "keep_original_shape", "tiling_size", "lvae_num_samples",
        "downsamp", "fill_factor", "dark_frame_context_length",
    },
}


class ApplyOverrides(BaseModel):
    """Sparse user-authored overrides for the `apply` section."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointOverrides | None = None
    input: ApplyInputOverrides | None = None
    inference: ApplyInferenceOverrides | None = None
    postprocess: ApplyPostprocessOverrides | None = None
    saving: ApplySavingOverrides | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_layout(cls, value):
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        migrated: dict[str, Any] = {}

        for section, keys in _APPLY_LEGACY_GROUPS.items():
            nested = raw.pop(section, None)
            if nested is not None:
                migrated[section] = dict(nested)
            for key in keys:
                if key in raw:
                    if section in migrated and key in migrated[section]:
                        raise ValueError(f"apply config defines both '{key}' and '{section}.{key}'.")
                    migrated.setdefault(section, {})[key] = raw.pop(key)

        postprocess = raw.pop("postprocess", None)
        if postprocess is not None:
            migrated["postprocess"] = dict(postprocess)
        if "denormalize_output" in raw:
            migrated.setdefault("postprocess", {})["denormalize"] = raw.pop("denormalize_output")
        legacy_color = raw.pop("color_code_prm", None)
        if legacy_color is not None:
            migrated.setdefault("postprocess", {}).setdefault("color_code", {}).update(dict(legacy_color))
        if "apply_color_code" in raw:
            migrated.setdefault("postprocess", {}).setdefault("color_code", {})["enabled"] = raw.pop("apply_color_code")

        saving = raw.pop("saving", None)
        legacy_output = raw.pop("output", None)
        if saving is not None and legacy_output is not None:
            raise ValueError("apply config cannot define both 'saving' and legacy 'output'.")
        if saving is not None:
            migrated["saving"] = dict(saving)
        elif legacy_output is not None:
            migrated["saving"] = dict(legacy_output)
        if "lvae_save_samples" in raw:
            migrated.setdefault("saving", {})["lvae_save_samples"] = raw.pop("lvae_save_samples")

        migrated.update(raw)
        return migrated


class EvaluateDataDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: str = Field(default="test", description=SPLIT_DESC)
    eval_gt: str | None = Field(default=None, description=EVAL_GT_DESC)
    overrides: dict[str, Any] | None = Field(default=None, description=DATA_PRM_UPDATE_DESC)
    limit_n_imgs: int | None = Field(default=None, gt=0, description=LIMIT_N_IMGS_DESC)
    timelapse_max: int | None = Field(default=None, gt=0, description=TIMELAPSE_MAX_DESC)


class EvaluateDataOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: str | None = Field(default=None, description=SPLIT_DESC)
    eval_gt: str | None = Field(default=None, description=EVAL_GT_DESC)
    overrides: dict[str, Any] | None = Field(default=None, description=DATA_PRM_UPDATE_DESC)
    limit_n_imgs: int | None = Field(default=None, gt=0, description=LIMIT_N_IMGS_DESC)
    timelapse_max: int | None = Field(default=None, gt=0, description=TIMELAPSE_MAX_DESC)


class EvaluateInferenceDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiling_size: TilingSizePolicy = Field(default="auto", description=TILING_SIZE_DESC)
    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    lvae_num_samples: int | None = Field(default=20, description=LVAE_NUM_SAMPLES_DESC)
    ch_out: int | None = Field(default=1, description=CH_OUT_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class EvaluateInferenceOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiling_size: TilingSizePolicy = Field(default=None, description=TILING_SIZE_DESC)
    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    lvae_num_samples: int | None = Field(default=None, description=LVAE_NUM_SAMPLES_DESC)
    ch_out: int | None = Field(default=None, description=CH_OUT_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class EvaluateSavingDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_folder: str | None = Field(default=None, description=EVALUATE_SAVE_FOLDER_DESC)
    overwrite: bool = Field(default=False, description=OVERWRITE_DESC)


class EvaluateSavingOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_folder: str | None = Field(default=None, description=EVALUATE_SAVE_FOLDER_DESC)
    overwrite: bool | None = Field(default=None, description=OVERWRITE_DESC)


class EvaluateDefaults(BaseModel):
    """Canonical LISAI defaults for the `evaluate` inference section."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointDefaults = Field(default_factory=CheckpointDefaults)
    data: EvaluateDataDefaults = Field(default_factory=EvaluateDataDefaults)
    inference: EvaluateInferenceDefaults = Field(default_factory=EvaluateInferenceDefaults)
    metrics: list[str] | None = Field(default=None, description=METRICS_LIST_DESC)
    saving: EvaluateSavingDefaults = Field(default_factory=EvaluateSavingDefaults)


_EVALUATE_LEGACY_GROUPS = {
    "checkpoint": {"best_or_last", "epoch_number"},
    "data": {"split", "eval_gt", "limit_n_imgs", "timelapse_max"},
    "inference": {"tiling_size", "crop_size", "lvae_num_samples", "ch_out"},
    "saving": {"save_folder", "overwrite"},
}


class EvaluateOverrides(BaseModel):
    """Sparse user-authored overrides for the `evaluate` section."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: CheckpointOverrides | None = None
    data: EvaluateDataOverrides | None = None
    inference: EvaluateInferenceOverrides | None = None
    metrics: list[str] | None = Field(default=None, description=METRICS_LIST_DESC)
    saving: EvaluateSavingOverrides | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_layout(cls, value):
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        migrated: dict[str, Any] = {}

        for section, keys in _EVALUATE_LEGACY_GROUPS.items():
            nested = raw.pop(section, None)
            if nested is not None:
                migrated[section] = dict(nested)
            for key in keys:
                if key in raw:
                    if section in migrated and key in migrated[section]:
                        raise ValueError(f"evaluate config defines both '{key}' and '{section}.{key}'.")
                    migrated.setdefault(section, {})[key] = raw.pop(key)

        if "metrics_list" in raw:
            if "metrics" in raw:
                raise ValueError("evaluate config cannot define both 'metrics' and legacy 'metrics_list'.")
            migrated["metrics"] = raw.pop("metrics_list")
        elif "metrics" in raw:
            migrated["metrics"] = raw.pop("metrics")

        if "data_prm_update" in raw:
            migrated.setdefault("data", {})["overrides"] = raw.pop("data_prm_update")

        migrated.update(raw)
        return migrated


# Backward-compatible alias for code that still imports the previous name.
ApplyOutputOverrides = ApplySavingOverrides


__all__ = [
    "ApplyDefaults",
    "ApplyInferenceDefaults",
    "ApplyInferenceOverrides",
    "ApplyInputDefaults",
    "ApplyInputOverrides",
    "ApplyOutputMode",
    "ApplyOutputOverrides",
    "ApplyOverrides",
    "ApplyPostprocessDefaults",
    "ApplyPostprocessOverrides",
    "ApplySavingDefaults",
    "ApplySavingOverrides",
    "CheckpointDefaults",
    "CheckpointOverrides",
    "CheckpointSelector",
    "ColorCodeDefaults",
    "ColorCodeOverrides",
    "EvaluateDataDefaults",
    "EvaluateDataOverrides",
    "EvaluateDefaults",
    "EvaluateInferenceDefaults",
    "EvaluateInferenceOverrides",
    "EvaluateOverrides",
    "EvaluateSavingDefaults",
    "EvaluateSavingOverrides",
    "PositiveTilingSize",
    "SaveInputMode",
    "TilingSizePolicy",
]
