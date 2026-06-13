from __future__ import annotations

import json
from pathlib import Path

from lisai.config import load_yaml, settings
from lisai.config.json_schema import preprocess_json_schema, write_preprocess_json_schema
from lisai.preprocess.core import PreprocessConfig
from lisai.preprocess.pipelines import PIPELINES_REGISTRY


def _pipeline_supported_values(pipeline_cls: type, field_name: str, fallback_attr: str) -> list[str]:
    values = getattr(pipeline_cls, field_name, {getattr(pipeline_cls, fallback_attr, None)})
    return sorted(str(value) for value in values if value is not None)


def _condition_for_pipeline(schema: dict, pipeline_name: str) -> dict:
    for condition in schema.get("allOf", []):
        pipeline_property = condition.get("if", {}).get("properties", {}).get("pipeline", {})
        if pipeline_property.get("const") == pipeline_name:
            return condition
    raise AssertionError(f"Did not find preprocess schema condition for pipeline '{pipeline_name}'.")


def test_preprocess_yaml_example_validates_against_preprocess_config_model():
    cfg = load_yaml(Path("configs/preprocess/preprocess.yml"))

    validated = PreprocessConfig.model_validate(cfg)

    assert validated.dataset_name
    assert validated.pipeline


def test_preprocess_json_schema_exposes_described_pipeline_specific_branches():
    schema = preprocess_json_schema()

    assert schema["properties"]["pipeline"]["enum"] == sorted(PIPELINES_REGISTRY)
    assert schema["properties"]["data_type"]["enum"] == sorted(settings.data_cfg.data_types)
    assert schema["properties"]["fmt"]["enum"] == sorted(settings.data_cfg.format)
    assert "allOf" in schema
    assert len(schema["allOf"]) == len(PIPELINES_REGISTRY)

    condition_names = {
        condition["if"]["properties"]["pipeline"]["const"]
        for condition in schema["allOf"]
    }
    assert condition_names == set(PIPELINES_REGISTRY)

    for pipeline_name, pipeline_cls in PIPELINES_REGISTRY.items():
        condition = _condition_for_pipeline(schema, pipeline_name)
        then_properties = condition["then"]["properties"]

        expected_data_types = _pipeline_supported_values(
            pipeline_cls,
            field_name="supported_data_types",
            fallback_attr="data_type",
        )
        data_type_property = then_properties["data_type"]
        if len(expected_data_types) == 1:
            assert data_type_property["const"] == expected_data_types[0]
        else:
            assert data_type_property["enum"] == expected_data_types

        expected_fmts = _pipeline_supported_values(
            pipeline_cls,
            field_name="supported_fmts",
            fallback_attr="fmt",
        )
        fmt_property = then_properties["fmt"]
        if len(expected_fmts) == 1:
            assert fmt_property["const"] == expected_fmts[0]
        else:
            assert fmt_property["enum"] == expected_fmts

    shared_properties = schema["properties"]
    single_condition = _condition_for_pipeline(schema, "single_recon")
    pipeline_cfg_properties = single_condition["then"]["properties"]["pipeline_cfg"]["properties"]

    assert shared_properties["dataset_name"]["description"]
    assert shared_properties["pipeline"]["description"]
    assert shared_properties["registry"]["$ref"] == "#/$defs/PreprocessRegistryConfig"
    assert shared_properties["split"]["$ref"] == "#/$defs/PreprocessSplitConfig"
    assert pipeline_cfg_properties["dump_subfolder"]["description"]
    assert pipeline_cfg_properties["crop_size"]["description"]
    assert schema["$defs"]["PreprocessRegistryDefaultsConfig"]["properties"]["eval_gt"]["description"]
    assert schema["$defs"]["PreprocessSplitConfig"]["properties"]["mode"]["description"]
    assert schema["$defs"]["PreprocessLogConfig"]["properties"]["enabled"]["description"]


def test_write_preprocess_json_schema_writes_json_file(tmp_path: Path):
    output_path = tmp_path / "preprocess.schema.json"

    written_path = write_preprocess_json_schema(output_path)

    assert written_path == output_path
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["title"] == "PreprocessConfig"
    assert data["allOf"]
