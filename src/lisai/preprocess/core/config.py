# lisai/data/preprocess/config.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetUsage = Literal["training", "evaluation"]
SplitMode = Literal["random", "manual", "reuse"]
SplitMatchBy = Literal["source_name", "source_relpath", "sample_id"]


class PreprocessLogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether to write a YAML preprocess manifest for this run.",
    )


class PreprocessSplitRandomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(
        default=0,
        description="Random seed used to shuffle items before assigning validation and test splits.",
    )
    val_fraction: float = Field(
        default=0.1,
        description="Fraction of source items assigned to the validation split.",
    )
    test_fraction: float = Field(
        default=0.1,
        description="Fraction of source items assigned to the test split.",
    )

    @field_validator("val_fraction", "test_fraction")
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if value < 0 or value >= 1:
            raise ValueError("Split fractions must be in the interval [0, 1).")
        return value

    @model_validator(mode="after")
    def _validate_total_fraction(self):
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("Validation and test fractions must sum to less than 1.")
        return self


class PreprocessSplitManualConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_by: SplitMatchBy = Field(
        default="source_name",
        description=(
            "Identifier style used to match manual split entries against source items. "
            "Use source_name for incoming filenames, source_relpath for paths relative to the dump root, "
            "or sample_id for generated preprocess names."
        ),
    )
    val: list[str] = Field(
        default_factory=list,
        description="Items that should be assigned to the validation split.",
    )
    test: list[str] = Field(
        default_factory=list,
        description="Items that should be assigned to the test split.",
    )

    @model_validator(mode="after")
    def _validate_no_overlap(self):
        overlap = sorted(set(self.val) & set(self.test))
        if overlap:
            raise ValueError(f"Manual split entries cannot appear in both val and test: {overlap}")
        return self


class PreprocessSplitReuseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str | None = Field(
        default=None,
        description="Optional path to a previous preprocess YAML manifest whose split assignments should be reused.",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Dataset name of a previous preprocess run to reuse split assignments from.",
    )
    data_type: Literal["raw", "recon"] | None = Field(
        default=None,
        description="Data type of the previous preprocess run when reusing a split by dataset name.",
    )
    match_by: SplitMatchBy = Field(
        default="source_relpath",
        description="Identifier style used to match current source items to entries stored in the reused manifest.",
    )

    @model_validator(mode="after")
    def _validate_source(self):
        if self.manifest_path is None and self.dataset_name is None:
            raise ValueError("Reuse split requires either `manifest_path` or `dataset_name`.")
        return self


class PreprocessSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Whether to assign saved preprocess outputs to train, validation, and test folders.",
    )
    mode: SplitMode = Field(
        default="random",
        description="Split strategy to use when split.enabled is true: random, manual, or reuse.",
    )
    random: PreprocessSplitRandomConfig = Field(
        default_factory=PreprocessSplitRandomConfig,
        description="Parameters for random split assignment.",
    )
    manual: PreprocessSplitManualConfig = Field(
        default_factory=PreprocessSplitManualConfig,
        description="Parameters for manually selecting validation and test items.",
    )
    reuse: PreprocessSplitReuseConfig | None = Field(
        default=None,
        description="Parameters for reusing split assignments from a previous preprocess manifest.",
    )

    @model_validator(mode="after")
    def _validate_selected_mode(self):
        if not self.enabled:
            return self
        if self.mode == "reuse" and self.reuse is None:
            raise ValueError("`split.reuse` must be provided when `split.mode='reuse'`.")
        return self


class PreprocessRegistryDefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str | None = Field(
        default=None,
        description="Optional registry default input path/key override for this preprocess data type.",
    )
    target: str | None = Field(
        default=None,
        description="Optional registry default supervised target path/key override. Use null when no default should be selected.",
    )
    eval_gt: str | None = Field(
        default=None,
        description="Optional registry default ground-truth path/key used for evaluation. Use null when no default should be selected.",
    )


class PreprocessRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: PreprocessRegistryDefaultsConfig = Field(
        default_factory=PreprocessRegistryDefaultsConfig,
        description="Optional registry default overrides merged over defaults inferred from produced outputs.",
    )


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(
        description="Dataset name as registered in the project data paths and registry.",
    )
    pipeline: str = Field(
        description="Preprocess pipeline to run. This selects the valid schema for pipeline_cfg.",
    )
    data_type: str = Field(
        description="Dataset data type produced by preprocessing, typically raw or recon.",
    )
    fmt: str = Field(
        description="Output data format used for canonical naming and saving templates.",
    )
    usage: DatasetUsage = Field(
        default="training",
        description="Intended dataset usage in the registry: training datasets may also be evaluated, evaluation datasets are test-only.",
    )
    pipeline_cfg: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline-specific configuration. Editor hints depend on the selected pipeline.",
    )
    log: PreprocessLogConfig = Field(
        default_factory=PreprocessLogConfig,
        description="Options controlling YAML manifest logging for the preprocess run.",
    )
    registry: PreprocessRegistryConfig = Field(
        default_factory=PreprocessRegistryConfig,
        description="Options controlling metadata written to the dataset registry.",
    )
    split: PreprocessSplitConfig = Field(
        default_factory=PreprocessSplitConfig,
        description="Options controlling optional train, validation, and test split assignment.",
    )
