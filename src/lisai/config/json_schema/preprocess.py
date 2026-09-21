from __future__ import annotations

import copy
import json
from dataclasses import fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from lisai.config import settings
from lisai.preprocess.core import PreprocessConfig
from lisai.preprocess.core.config import (
    PreprocessLogConfig,
    PreprocessRegistryConfig,
    PreprocessRegistryDefaultsConfig,
    PreprocessSplitConfig,
    PreprocessSplitManualConfig,
    PreprocessSplitRandomConfig,
    PreprocessSplitReuseConfig,
)
from lisai.preprocess.pipelines import PIPELINES_REGISTRY

PREPROCESS_MODEL_TYPES: tuple[type[BaseModel], ...] = (
    PreprocessConfig,
    PreprocessLogConfig,
    PreprocessRegistryConfig,
    PreprocessRegistryDefaultsConfig,
    PreprocessSplitConfig,
    PreprocessSplitRandomConfig,
    PreprocessSplitManualConfig,
    PreprocessSplitReuseConfig,
)


def _ensure_model_field_descriptions() -> None:
    missing: dict[str, list[str]] = {}
    for model_type in PREPROCESS_MODEL_TYPES:
        missing_fields = [
            name for name, model_field in model_type.model_fields.items() if not (model_field.description or "").strip()
        ]
        if missing_fields:
            missing[model_type.__name__] = missing_fields

    if missing:
        details = ", ".join(f"{model}: {fields}" for model, fields in missing.items())
        raise ValueError(f"Missing preprocess schema descriptions for: {details}")


def _ensure_pipeline_cfg_descriptions(pipeline_name: str, cfg_type: type[Any]) -> None:
    if not is_dataclass(cfg_type):
        raise TypeError(f"Pipeline '{pipeline_name}' Config must be a dataclass type.")

    missing = [field.name for field in dataclass_fields(cfg_type) if not str(field.metadata.get("description", "")).strip()]
    if missing:
        raise ValueError(f"Pipeline '{pipeline_name}' is missing config field descriptions for: {missing}")


def _merge_defs(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        existing = target.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"Conflicting preprocess schema definition for '{key}'.")
        target[key] = value


def _pipeline_cfg_schema(pipeline_name: str, pipeline_cls: type[Any]) -> dict[str, Any]:
    cfg_type = getattr(pipeline_cls, "Config", None)
    if cfg_type is None:
        raise TypeError(f"Pipeline '{pipeline_name}' does not define a Config dataclass.")

    _ensure_pipeline_cfg_descriptions(pipeline_name, cfg_type)
    schema = TypeAdapter(cfg_type).json_schema()
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    return schema


def _pipeline_supported_fmts(pipeline_name: str, pipeline_cls: type[Any]) -> list[str]:
    supported_fmts = getattr(
        pipeline_cls,
        "supported_fmts",
        {getattr(pipeline_cls, "fmt", None)},
    )
    fmts = sorted(str(fmt) for fmt in supported_fmts if fmt is not None)
    if not fmts:
        raise ValueError(f"Pipeline '{pipeline_name}' does not declare any supported formats.")
    return fmts


def _pipeline_supported_data_types(pipeline_name: str, pipeline_cls: type[Any]) -> list[str]:
    supported_data_types = getattr(
        pipeline_cls,
        "supported_data_types",
        {getattr(pipeline_cls, "data_type", None)},
    )
    data_types = sorted(str(data_type) for data_type in supported_data_types if data_type is not None)
    if not data_types:
        raise ValueError(f"Pipeline '{pipeline_name}' does not declare any supported data types.")
    return data_types


def _pipeline_constrained_property(
    *,
    base_property: dict[str, Any],
    pipeline_name: str,
    allowed_values: list[str],
) -> dict[str, Any]:
    constrained = copy.deepcopy(base_property)
    constrained.pop("enum", None)
    if len(allowed_values) == 1:
        constrained["const"] = allowed_values[0]
        constrained["default"] = allowed_values[0]
    else:
        constrained["enum"] = allowed_values

    description = str(constrained.get("description", "")).strip()
    allowed = ", ".join(allowed_values)
    constrained["description"] = f"{description} Allowed for pipeline '{pipeline_name}': {allowed}."
    return constrained


def preprocess_json_schema() -> dict:
    _ensure_model_field_descriptions()

    base_schema = PreprocessConfig.model_json_schema()
    pipeline_names = sorted(PIPELINES_REGISTRY)
    available_data_types = sorted(settings.data_cfg.data_types)
    available_fmts = sorted(settings.data_cfg.format)
    properties = copy.deepcopy(base_schema["properties"])
    pipeline_property = copy.deepcopy(base_schema["properties"]["pipeline"])
    pipeline_property["enum"] = pipeline_names
    data_type_property = copy.deepcopy(base_schema["properties"]["data_type"])
    data_type_property["enum"] = available_data_types
    fmt_property = copy.deepcopy(base_schema["properties"]["fmt"])
    fmt_property["enum"] = available_fmts
    pipeline_cfg_description = str(base_schema["properties"]["pipeline_cfg"].get("description", "")).strip()
    required = list(base_schema.get("required", []))
    defs = copy.deepcopy(base_schema.get("$defs", {}))

    conditions: list[dict[str, Any]] = []
    for pipeline_name, pipeline_cls in sorted(PIPELINES_REGISTRY.items()):
        supported_data_types = _pipeline_supported_data_types(pipeline_name, pipeline_cls)
        supported_fmts = _pipeline_supported_fmts(pipeline_name, pipeline_cls)

        unknown_data_types = sorted(set(supported_data_types) - set(available_data_types))
        if unknown_data_types:
            raise ValueError(
                f"Pipeline '{pipeline_name}' declares unsupported data types {unknown_data_types}. "
                f"Known data types from data_config are: {available_data_types}"
            )
        unknown_fmts = sorted(set(supported_fmts) - set(available_fmts))
        if unknown_fmts:
            raise ValueError(
                f"Pipeline '{pipeline_name}' declares unsupported formats {unknown_fmts}. "
                f"Known formats from data_config are: {available_fmts}"
            )

        pipeline_cfg_schema = _pipeline_cfg_schema(pipeline_name, pipeline_cls)
        _merge_defs(defs, pipeline_cfg_schema.pop("$defs", {}))
        pipeline_cfg_schema.setdefault("description", pipeline_cfg_description)

        constrained_data_type = _pipeline_constrained_property(
            base_property=data_type_property,
            pipeline_name=pipeline_name,
            allowed_values=supported_data_types,
        )
        constrained_fmt = _pipeline_constrained_property(
            base_property=fmt_property,
            pipeline_name=pipeline_name,
            allowed_values=supported_fmts,
        )
        conditions.append(
            {
                "if": {
                    "properties": {
                        "pipeline": {"const": pipeline_name},
                    },
                    "required": ["pipeline"],
                },
                "then": {
                    "properties": {
                        "data_type": constrained_data_type,
                        "fmt": constrained_fmt,
                        "pipeline_cfg": pipeline_cfg_schema,
                    },
                },
            }
        )

    properties["pipeline"] = pipeline_property
    properties["data_type"] = data_type_property
    properties["fmt"] = fmt_property

    schema: dict[str, Any] = {
        "title": base_schema.get("title", "PreprocessConfig"),
        "type": "object",
        "additionalProperties": bool(base_schema.get("additionalProperties", False)),
        "properties": properties,
        "required": required,
        "allOf": conditions,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def write_preprocess_json_schema(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = preprocess_json_schema()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path
