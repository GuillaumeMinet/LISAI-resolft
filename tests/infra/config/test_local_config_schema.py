from __future__ import annotations

import json
from pathlib import Path

from lisai.config.json_schema import local_config_json_schema, write_local_config_json_schema


def test_local_config_schema_exposes_output_policy_choices():
    schema = local_config_json_schema()
    output = schema["$defs"]["LocalInferenceConfig"]["properties"]["output"]
    output_schema = schema["$defs"]["LocalInferenceOutputConfig"]["properties"]

    assert output["$ref"] == "#/$defs/LocalInferenceOutputConfig"
    assert output_schema["mode"]["enum"] == [
        "default",
        "in_place",
        "folder_inside",
        "folder_outside",
    ]
    assert output_schema["save_input_mode"]["enum"] == [
        "always",
        "never",
        "if_not_in_place",
    ]


def test_local_config_schema_exposes_console_progress_preference():
    schema = local_config_json_schema()
    console = schema["properties"]["console"]
    progress_bar = schema["$defs"]["LocalConsoleConfig"]["properties"]["progress_bar"]

    assert console["$ref"] == "#/$defs/LocalConsoleConfig"
    assert progress_bar["default"] is None
    assert {"type": "boolean"} in progress_bar["anyOf"]
    assert {"type": "null"} in progress_bar["anyOf"]


def test_write_local_config_json_schema_writes_json_file(tmp_path: Path):
    output_path = tmp_path / "local-config.schema.json"

    written_path = write_local_config_json_schema(output_path)

    assert written_path == output_path
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["title"] == "LocalConfig"
