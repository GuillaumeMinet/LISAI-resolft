"""Explicit creation of user-owned preprocess configs from packaged templates.

Templates are inert examples, never runnable directly. User dataset-specific YAMLs
live under the selected writable experiment config root, not package resources.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from lisai.config.io.config_paths import ConfigPathResolver, managed_output_path
from lisai.config.io.schema_hints import schema_header
from lisai.config.io.yaml import load_yaml
from lisai.config.settings import Settings

from .core.config import PreprocessConfig
from .pipelines import PIPELINES_REGISTRY


def create_preprocess_config_from_template(
    template: str,
    *,
    dataset_name: str | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
    stg: Settings,
) -> tuple[Path, list[str]]:
    """Copy a generic package template to the flat writable preprocess root.

    A missing dataset_name leaves a conspicuous placeholder, which cannot be run
    until replaced. If a name is supplied, validate the resulting config against
    its pipeline's dataclass before saving it.
    """
    if not template or Path(template).is_absolute() or ".." in Path(template).parts:
        raise ValueError("Choose a packaged preprocess template name, not a file path")
    name = template.removeprefix("templates/")
    reference = ConfigPathResolver("preprocess", stg=stg).resolve_reference(
        f"templates/{name}"
    )
    if reference is None or reference.ownership != "built-in":
        raise ValueError("Preprocess templates must be package-owned resources")
    body = load_yaml(reference.location)
    if not isinstance(body, dict):
        raise ValueError(f"Invalid preprocess template: {template}")
    if body.get("pipeline") not in PIPELINES_REGISTRY:
        raise ValueError(f"Unknown template pipeline: {body.get('pipeline')!r}")
    warnings: list[str] = []
    if dataset_name is None:
        body["dataset_name"] = "CHANGEME"
        warnings.append("dataset_name is CHANGEME; edit before preprocessing.")
    elif not dataset_name.strip() or dataset_name.strip() == "CHANGEME":
        raise ValueError("Provide a real --dataset name, not CHANGEME")
    else:
        body["dataset_name"] = dataset_name.strip()
        validated = PreprocessConfig.model_validate(body)
        PIPELINES_REGISTRY[validated.pipeline].parse_cfg(validated.pipeline_cfg)

    filename = body["dataset_name"] if dataset_name is not None else Path(name).stem
    # Dataset names are not filenames: reject separators/traversal for the default.
    if output is None and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", filename)
                           or filename in {".", ".."}):
        raise ValueError("Invalid dataset name for default output filename; pass --output")
    target = managed_output_path(output if output is not None else f"{filename}.yml",
                                 root=stg.PREPROCESS_CONFIG_DIR, kind="preprocess")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Preprocess config already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    hint = schema_header(target, stg.CONFIGS_ROOT / "schema", "preprocess.schema.json")
    header = hint + "\n" if hint else ""
    target.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    if not (stg.CONFIGS_ROOT / "schema" / "preprocess.schema.json").is_file():
        warnings.append("Editor schemas missing; run `lisai configs schema refresh`.")
    return target, warnings
