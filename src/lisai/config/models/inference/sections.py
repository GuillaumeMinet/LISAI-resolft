from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ApplyOutputMode: TypeAlias = Literal["default", "in_place", "folder_inside", "folder_outside"]
SaveInputMode: TypeAlias = Literal["always", "never", "if_not_in_place"]

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
TIMELAPSE_MAX_DESC = "Optional maximum number of timelapse frames to process during apply."
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
RESULTS_DESC = "Optional in-memory metrics accumulator used by the Python API when evaluating programmatically."
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
LIMIT_N_IMGS_DESC = "Optional cap on the number of images or batches evaluated."


CheckpointSelector = Literal["best", "last", "both"]
PositiveTilingSize: TypeAlias = Annotated[int, Field(gt=0)]
TilingSizePolicy: TypeAlias = PositiveTilingSize | Literal["auto", "off"] | None


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


class ColorCodeDefaults(BaseModel):
    """Complete color-coding settings used for volumetric apply outputs."""

    model_config = ConfigDict(extra="forbid")

    colormap: str = Field(default="turbo", description=COLORMAP_DESC)
    saturation: float = Field(default=0.35, description=SATURATION_DESC)
    add_colorbar: bool = Field(default=True, description=ADD_COLORBAR_DESC)
    zstep: float = Field(default=0.4, description=ZSTEP_DESC)


class ColorCodeOverrides(BaseModel):
    """Sparse user overrides for volumetric color-coding settings."""

    model_config = ConfigDict(extra="forbid")

    colormap: str | None = Field(default=None, description=COLORMAP_DESC)
    saturation: float | None = Field(default=None, description=SATURATION_DESC)
    add_colorbar: bool | None = Field(default=None, description=ADD_COLORBAR_DESC)
    zstep: float | None = Field(default=None, description=ZSTEP_DESC)


class ApplyDefaults(BaseModel):
    """Fully resolved defaults for the `apply` inference section."""

    model_config = ConfigDict(extra="forbid")

    epoch_number: int | None = Field(default=None, description=EPOCH_NUMBER_DESC)
    best_or_last: CheckpointSelector = Field(default="best", description=BEST_OR_LAST_DESC)
    filters: list[str] = Field(default_factory=lambda: ["tiff", "tif"], description=FILTERS_DESC)
    skip_if_contain: list[str] | None = Field(default=None, description=SKIP_IF_CONTAIN_DESC)
    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    keep_original_shape: bool = Field(default=True, description=KEEP_ORIGINAL_SHAPE_DESC)
    tiling_size: TilingSizePolicy = Field(default="auto", description=TILING_SIZE_DESC)
    stack_selection_idx: int | None = Field(default=None, description=STACK_SELECTION_IDX_DESC)
    timelapse_max: int | None = Field(default=None, description=TIMELAPSE_MAX_DESC)
    lvae_num_samples: int | None = Field(default=20, description=LVAE_NUM_SAMPLES_DESC)
    lvae_save_samples: bool = Field(default=True, description=LVAE_SAVE_SAMPLES_DESC)
    denormalize_output: bool = Field(default=True, description=DENORMALIZE_OUTPUT_DESC)
    downsamp: int | None = Field(default=None, description=DOWNSAMP_DESC)
    fill_factor: float | None = Field(default=None, gt=0, le=1, description=FILL_FACTOR_DESC)
    apply_color_code: bool = Field(default=False, description=APPLY_COLOR_CODE_DESC)
    color_code_prm: ColorCodeDefaults = Field(default_factory=ColorCodeDefaults, description="Nested volumetric color-coding settings used when apply_color_code is true.")
    dark_frame_context_length: bool = Field(default=False, description=DARK_FRAME_CONTEXT_LENGTH_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class ApplyOutputOverrides(BaseModel):
    """Optional output-policy overrides for one inference workflow."""

    model_config = ConfigDict(extra="forbid")

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
        selected = [
            self.mode is not None,
            self.save_folder is not None,
            self.in_place is not None,
        ]
        if sum(selected) > 1:
            raise ValueError(
                "apply.output.mode, apply.output.save_folder, and apply.output.in_place "
                "are mutually exclusive."
            )
        return self


class ApplyOverrides(BaseModel):
    """Sparse user-authored overrides for the `apply` section."""

    model_config = ConfigDict(extra="forbid")

    output: ApplyOutputOverrides | None = Field(
        default=None,
        description="Optional output-policy override for this inference config.",
    )
    epoch_number: int | None = Field(default=None, description=EPOCH_NUMBER_DESC)
    best_or_last: CheckpointSelector | None = Field(default=None, description=BEST_OR_LAST_DESC)
    filters: list[str] | None = Field(default=None, description=FILTERS_DESC)
    skip_if_contain: list[str] | None = Field(default=None, description=SKIP_IF_CONTAIN_DESC)
    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    keep_original_shape: bool | None = Field(default=None, description=KEEP_ORIGINAL_SHAPE_DESC)
    tiling_size: TilingSizePolicy = Field(default=None, description=TILING_SIZE_DESC)
    stack_selection_idx: int | None = Field(default=None, description=STACK_SELECTION_IDX_DESC)
    timelapse_max: int | None = Field(default=None, description=TIMELAPSE_MAX_DESC)
    lvae_num_samples: int | None = Field(default=None, description=LVAE_NUM_SAMPLES_DESC)
    lvae_save_samples: bool | None = Field(default=None, description=LVAE_SAVE_SAMPLES_DESC)
    denormalize_output: bool | None = Field(default=None, description=DENORMALIZE_OUTPUT_DESC)
    downsamp: int | None = Field(default=None, description=DOWNSAMP_DESC)
    fill_factor: float | None = Field(default=None, gt=0, le=1, description=FILL_FACTOR_DESC)
    apply_color_code: bool | None = Field(default=None, description=APPLY_COLOR_CODE_DESC)
    color_code_prm: ColorCodeOverrides | None = Field(default=None, description="Nested volumetric color-coding overrides used when apply_color_code is true.")
    dark_frame_context_length: bool | None = Field(default=None, description=DARK_FRAME_CONTEXT_LENGTH_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class EvaluateDefaults(BaseModel):
    """Fully resolved defaults for the `evaluate` inference section."""

    model_config = ConfigDict(extra="forbid")

    best_or_last: CheckpointSelector = Field(default="best", description=BEST_OR_LAST_DESC)
    epoch_number: int | None = Field(default=None, description=EPOCH_NUMBER_DESC)
    tiling_size: TilingSizePolicy = Field(default="auto", description=TILING_SIZE_DESC)
    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    metrics_list: list[str] | None = Field(default=None, description=METRICS_LIST_DESC)
    lvae_num_samples: int | None = Field(default=20, description=LVAE_NUM_SAMPLES_DESC)
    results: dict[str, Any] | None = Field(default=None, description=RESULTS_DESC)
    save_folder: str | None = Field(default=None, description=EVALUATE_SAVE_FOLDER_DESC)
    overwrite: bool = Field(default=False, description=OVERWRITE_DESC)
    eval_gt: str | None = Field(default=None, description=EVAL_GT_DESC)
    data_prm_update: dict[str, Any] | None = Field(default=None, description=DATA_PRM_UPDATE_DESC)
    ch_out: int | None = Field(default=1, description=CH_OUT_DESC)
    split: str = Field(default="test", description=SPLIT_DESC)
    limit_n_imgs: int | None = Field(default=None, gt=0, description=LIMIT_N_IMGS_DESC)
    timelapse_max: int | None = Field(default=None, gt=0, description=TIMELAPSE_MAX_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


class EvaluateOverrides(BaseModel):
    """Sparse user-authored overrides for the `evaluate` section."""

    model_config = ConfigDict(extra="forbid")

    best_or_last: CheckpointSelector | None = Field(default=None, description=BEST_OR_LAST_DESC)
    epoch_number: int | None = Field(default=None, description=EPOCH_NUMBER_DESC)
    tiling_size: TilingSizePolicy = Field(default=None, description=TILING_SIZE_DESC)
    crop_size: int | tuple[int, int] | None = Field(default=None, description=CROP_SIZE_DESC)
    metrics_list: list[str] | None = Field(default=None, description=METRICS_LIST_DESC)
    lvae_num_samples: int | None = Field(default=None, description=LVAE_NUM_SAMPLES_DESC)
    save_folder: str | None = Field(default=None, description=EVALUATE_SAVE_FOLDER_DESC)
    overwrite: bool | None = Field(default=None, description=OVERWRITE_DESC)
    eval_gt: str | None = Field(default=None, description=EVAL_GT_DESC)
    data_prm_update: dict[str, Any] | None = Field(default=None, description=DATA_PRM_UPDATE_DESC)
    ch_out: int | None = Field(default=None, description=CH_OUT_DESC)
    split: str | None = Field(default=None, description=SPLIT_DESC)
    limit_n_imgs: int | None = Field(default=None, gt=0, description=LIMIT_N_IMGS_DESC)
    timelapse_max: int | None = Field(default=None, gt=0, description=TIMELAPSE_MAX_DESC)

    @field_validator("tiling_size", mode="before")
    @classmethod
    def _normalize_tiling_size(cls, value):
        return _normalize_tiling_size_policy(value)


__all__ = [
    "ColorCodeDefaults",
    "ColorCodeOverrides",
    "PositiveTilingSize",
    "TilingSizePolicy",
    "ApplyOutputOverrides",
    "ApplyDefaults",
    "ApplyOverrides",
    "CheckpointSelector",
    "EvaluateDefaults",
    "EvaluateOverrides",
]
