from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from lisai.config.io.metadata import split_config_metadata, strip_config_metadata
from lisai.config.io.yaml import load_yaml
from lisai.config.models import ContinueTrainingConfig, ExperimentConfig, RetrainConfig

TrainingConfigKind = Literal["preset", "template", "example", "local"]
TrainingPresetName = str

KIND_TO_FOLDER: dict[TrainingConfigKind, str] = {
    "preset": "presets",
    "template": "templates",
    "example": "examples",
    "local": "local",
}
FOLDER_TO_KIND = {folder: kind for kind, folder in KIND_TO_FOLDER.items()}

PLACEHOLDER_RE = re.compile(r"^(?:<[A-Za-z_][A-Za-z0-9_.-]*>|\$\{[A-Za-z_][A-Za-z0-9_.-]*\})$")
TRAINING_SUFFIXES = (".yml", ".yaml")
EXPERIMENT_SCHEMA_LINE = "# yaml-language-server: $schema=../../schema/experiment.schema.json"
TEMPLATE_SCHEMA_LINE = "# yaml-language-server: $schema=../../schema/experiment-template.schema.json"

SECTION_HEADERS: dict[str, str] = {
    "metadata": "Catalog metadata",
    "experiment": "General experiment settings",
    "routing": "Paths",
    "data": "Dataset / DataLoader configuration",
    "normalization": "Normalization / Noise",
    "noise_model": "Noise model configuration",
    "model": "Model configuration",
    "training": "Training configuration",
    "loss_function": "Loss configuration",
    "saving": "Saving configuration",
    "tensorboard": "Tensorboard configuration",
    "load_model": "Model loading configuration",
    "recovery": "Recovery configuration",
}


class _IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow=flow, indentless=False)


@dataclass(frozen=True)
class TrainingConfigEntry:
    path: Path
    relative_path: str
    kind: TrainingConfigKind
    name: str
    task: str | None
    description: str | None
    metadata: dict[str, Any]


def training_config_root() -> Path:
    from lisai.config.settings import settings

    return settings.TRAINING_CONFIG_DIR


def resolve_training_config_path(config_ref: str | Path) -> Path:
    """Resolve an explicit training config path or training-root relative subpath."""
    root = training_config_root()
    config_path = Path(config_ref).expanduser()

    candidates = list(_candidate_paths(config_path))
    if not config_path.is_absolute():
        candidates.extend(_candidate_paths(root / config_path))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Training config not found: {config_ref}")


def resolve_training_preset_path(preset_ref: str | Path) -> Path:
    return _resolve_catalog_member_path(preset_ref, kind="preset")


def resolve_training_template_path(template_ref: str | Path) -> Path:
    return _resolve_catalog_member_path(template_ref, kind="template")


def discover_training_configs(
    *,
    kind: TrainingConfigKind | None = None,
    task: str | None = None,
) -> list[TrainingConfigEntry]:
    root = training_config_root()
    kinds = [kind] if kind is not None else list(KIND_TO_FOLDER)
    entries: list[TrainingConfigEntry] = []

    for current_kind in kinds:
        folder = root / KIND_TO_FOLDER[current_kind]
        if not folder.exists():
            continue
        for path in sorted(_iter_yaml_files(folder)):
            entry = _entry_from_path(path, root=root, fallback_kind=current_kind)
            if task is not None and entry.task != task:
                continue
            entries.append(entry)

    return entries


def load_training_config(config_ref: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None, Path]:
    path = resolve_training_config_path(config_ref)
    cfg = load_yaml(path)
    body, metadata = split_config_metadata(cfg)
    return body, metadata, path


def validate_training_config_dict(cfg: dict[str, Any]):
    body = strip_config_metadata(cfg)
    changeme_paths = _changeme_paths(body)
    if changeme_paths:
        joined = ", ".join(changeme_paths)
        raise ValueError(f"Training config still contains CHANGEME placeholder(s): {joined}")
    mode = _config_mode(body)
    return _authoring_model_for_mode(mode).model_validate(body)


def validate_training_template_dict(cfg: dict[str, Any]) -> None:
    body = strip_config_metadata(cfg)
    if not isinstance(body, dict):
        raise ValueError("Training template must be a YAML mapping.")

    unknown_roots = sorted(set(body) - _known_training_roots())
    if unknown_roots:
        roots = ", ".join(unknown_roots)
        raise ValueError(f"Unknown top-level template section(s): {roots}")

    invalid_placeholders: list[str] = []
    _collect_invalid_placeholders(body, path="", out=invalid_placeholders)
    if invalid_placeholders:
        joined = ", ".join(invalid_placeholders)
        raise ValueError(f"Invalid placeholder syntax at: {joined}")


def create_training_config_from_preset(
    preset: TrainingPresetName,
    *,
    exp_name: str | None = None,
    dataset_name: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    betaKL: float | None = None,
    loss: str | None = None,
    upsampling_factor: int | None = None,
    sampling_ratio: float | None = None,
    temporal_window: int | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, list[str]]:
    preset_path = resolve_training_preset_path(preset)
    preset_cfg = load_yaml(preset_path)
    cfg = strip_config_metadata(preset_cfg)
    warnings: list[str] = []

    if not exp_name:
        exp_name = "CHANGEME"
        warnings.append("experiment.exp_name was left as CHANGEME.")
    if not dataset_name:
        dataset_name = "CHANGEME"
        warnings.append("data.dataset_name was left as CHANGEME.")
    if input_name is None:
        input_name = ""
        warnings.append("data.input was left empty.")

    _dset(cfg, "experiment.exp_name", exp_name)
    _dset(cfg, "data.dataset_name", dataset_name)
    _dset(cfg, "data.input", input_name)

    task_cfg = _task_section(cfg)
    task_name = task_cfg.get("name")
    if not isinstance(task_name, str):
        raise ValueError(f"Preset must define `experiment.task.name`: {preset_path}")

    if task_name == "denoising_hdn":
        supervised = bool(task_cfg.get("supervised", False))
        if betaKL is not None:
            task_cfg["betaKL"] = float(betaKL)
        else:
            warnings.append("experiment.task.betaKL kept from the preset.")
        _dset(cfg, "data.paired", bool(supervised))
        _dset(cfg, "data.target", target_name if bool(supervised) else None)
        if bool(supervised) and not target_name:
            _dset(cfg, "data.target", "CHANGEME")
            warnings.append("data.target was left as CHANGEME for supervised denoising_hdn.")

    elif task_name in {"denoising_care", "denoising_unetrcan"}:
        if loss is not None:
            task_cfg["loss"] = loss
        if not target_name:
            target_name = "CHANGEME"
            warnings.append(f"data.target was left as CHANGEME for {task_name}.")
        _dset(cfg, "data.paired", True)
        _dset(cfg, "data.target", target_name)
        cfg.pop("loss_function", None)

    elif task_name == "upsamp_single_frame":
        if upsampling_factor is not None:
            task_cfg["upsampling_factor"] = int(upsampling_factor)
        else:
            warnings.append("experiment.task.upsampling_factor kept from the preset.")
        if sampling_ratio is not None:
            task_cfg["sampling_ratio"] = float(sampling_ratio)
        else:
            warnings.append("experiment.task.sampling_ratio kept from the preset.")
        _dset(cfg, "data.paired", False)
        _dset(cfg, "data.target", None)

    elif task_name == "upsamp_multiframes":
        if upsampling_factor is not None:
            task_cfg["upsampling_factor"] = int(upsampling_factor)
        else:
            warnings.append("experiment.task.upsampling_factor kept from the preset.")
        if temporal_window is not None:
            task_cfg["temporal_window"] = int(temporal_window)
        else:
            warnings.append("experiment.task.temporal_window kept from the preset.")
        _dset(cfg, "data.paired", False)
        _dset(cfg, "data.target", None)

    _validate_scaffold_config(cfg)
    output_path = _resolve_output_path(output=output, exp_name=exp_name)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_config_yaml(cfg, output_path)
    return output_path, warnings


def create_training_config_from_template(
    template: str | Path = "base_training",
    *,
    exp_name: str | None = None,
    dataset_name: str | None = None,
    input_name: str | None = None,
    target_name: str | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, list[str]]:
    template_path = resolve_training_template_path(template)
    template_cfg = load_yaml(template_path)
    validate_training_template_dict(template_cfg)
    cfg = _replace_template_placeholders(strip_config_metadata(template_cfg))
    warnings: list[str] = []

    if not exp_name:
        exp_name = "CHANGEME"
        warnings.append("experiment.exp_name was left as CHANGEME.")
    if not dataset_name:
        dataset_name = "CHANGEME"
        warnings.append("data.dataset_name was left as CHANGEME.")
    if input_name is None:
        input_name = ""
        warnings.append("data.input was left empty.")

    _dset(cfg, "experiment.exp_name", exp_name)
    _dset(cfg, "data.dataset_name", dataset_name)
    _dset(cfg, "data.input", input_name)
    if target_name is not None:
        _dset(cfg, "data.target", target_name)

    if _changeme_paths(cfg):
        warnings.append("Template placeholders were left as CHANGEME; edit them before validation or training.")

    _validate_scaffold_config(cfg)
    output_path = _resolve_output_path(output=output, exp_name=exp_name)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_config_yaml(cfg, output_path)
    return output_path, warnings


def save_training_config_yaml(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    metadata = cfg.get("metadata")
    is_template = isinstance(metadata, dict) and metadata.get("kind") == "template"
    schema_line = TEMPLATE_SCHEMA_LINE if is_template else EXPERIMENT_SCHEMA_LINE
    yaml_text = yaml.dump(cfg, Dumper=_IndentedSafeDumper, sort_keys=False)
    text = _with_section_comments(yaml_text, schema_line=schema_line)
    path.write_text(text, encoding="utf-8")


def _candidate_paths(path: Path) -> tuple[Path, ...]:
    if path.suffix:
        return (path,)
    return (path,) + tuple(path.with_suffix(suffix) for suffix in TRAINING_SUFFIXES)


def _resolve_catalog_member_path(config_ref: str | Path, *, kind: TrainingConfigKind) -> Path:
    folder_name = KIND_TO_FOLDER[kind]
    ref_path = Path(config_ref).expanduser()
    refs: list[Path]
    if ref_path.is_absolute() or (ref_path.parts and ref_path.parts[0] == folder_name):
        refs = [ref_path]
    else:
        refs = [Path(folder_name) / ref_path, ref_path]

    for ref in refs:
        try:
            path = resolve_training_config_path(ref)
        except FileNotFoundError:
            continue
        expected_root = (training_config_root() / folder_name).resolve()
        try:
            path.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError(f"{kind.title()} config must live under {expected_root}: {path}") from exc
        return path

    raise FileNotFoundError(f"{kind.title()} config not found: {config_ref}")


def _iter_yaml_files(root: Path):
    for suffix in TRAINING_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def _entry_from_path(path: Path, *, root: Path, fallback_kind: TrainingConfigKind) -> TrainingConfigEntry:
    metadata: dict[str, Any] = {}
    task = None
    description = None
    name = path.stem

    try:
        raw = load_yaml(path)
        body, parsed_metadata = split_config_metadata(raw)
        metadata = parsed_metadata or {}
        name = str(metadata.get("name") or name)
        task = metadata.get("task") or _infer_task_name(body)
        description = metadata.get("description")
        kind = metadata.get("kind") or fallback_kind
        if kind not in KIND_TO_FOLDER:
            kind = fallback_kind
    except Exception as exc:
        kind = fallback_kind
        description = f"Could not read config: {exc}"

    return TrainingConfigEntry(
        path=path.resolve(),
        relative_path=path.relative_to(root).as_posix(),
        kind=kind,
        name=name,
        task=str(task) if task is not None else None,
        description=str(description) if description is not None else None,
        metadata=metadata,
    )


def _infer_task_name(cfg: dict[str, Any]) -> str | None:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, dict):
        return None
    task = experiment.get("task")
    if isinstance(task, str):
        return task
    if isinstance(task, dict):
        name = task.get("name")
        return str(name) if name is not None else None
    return None


def _config_mode(cfg: dict[str, Any]) -> str:
    experiment = cfg.get("experiment")
    if isinstance(experiment, dict) and experiment.get("mode") is not None:
        mode = experiment["mode"]
    else:
        mode = cfg.get("mode", "train")
    if mode == "resume":
        mode = "continue_training"
    if mode not in {"train", "continue_training", "retrain"}:
        raise ValueError(f"Unknown mode: {mode}")
    return str(mode)


def _authoring_model_for_mode(mode: str):
    if mode == "train":
        return ExperimentConfig
    if mode == "continue_training":
        return ContinueTrainingConfig
    if mode == "retrain":
        return RetrainConfig
    raise ValueError(f"Unknown mode: {mode}")


def _known_training_roots() -> set[str]:
    roots = {"metadata", "mode"}
    for model in (ExperimentConfig, ContinueTrainingConfig, RetrainConfig):
        roots.update(model.model_fields)
    return roots


def _validate_scaffold_config(cfg: dict[str, Any]) -> None:
    body = strip_config_metadata(cfg)
    if _changeme_paths(body):
        validate_training_template_dict(body)
        return
    validate_training_config_dict(body)


def _replace_template_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_template_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_template_placeholders(item) for item in value]
    if isinstance(value, str) and _looks_like_placeholder(value):
        return "CHANGEME"
    return value


def _changeme_paths(value: Any, *, path: str = "") -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            out.extend(_changeme_paths(item, path=child_path))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for index, item in enumerate(value):
            out.extend(_changeme_paths(item, path=f"{path}[{index}]"))
        return out
    if value == "CHANGEME":
        return [path or "<root>"]
    return []


def _collect_invalid_placeholders(value: Any, *, path: str, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _collect_invalid_placeholders(item, path=child_path, out=out)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_invalid_placeholders(item, path=f"{path}[{index}]", out=out)
        return
    if isinstance(value, str) and _looks_like_placeholder(value) and not PLACEHOLDER_RE.match(value):
        out.append(path or "<root>")


def _looks_like_placeholder(value: str) -> bool:
    return (value.startswith("<") and value.endswith(">")) or (
        value.startswith("${") and value.endswith("}")
    )


def _task_section(cfg: dict[str, Any]) -> dict[str, Any]:
    experiment = cfg.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("`experiment` must be a mapping.")
    task = experiment.setdefault("task", {})
    if not isinstance(task, dict):
        task = {"name": task}
        experiment["task"] = task
    return task


def _dset(d: dict[str, Any], path: str, value: Any) -> None:
    cur = d
    keys = path.split(".")
    for key in keys[:-1]:
        child = cur.get(key)
        if not isinstance(child, dict):
            child = {}
            cur[key] = child
        cur = child
    cur[keys[-1]] = value


def _resolve_output_path(*, output: str | Path | None, exp_name: str) -> Path:
    root = training_config_root()
    if output is None:
        path = root / "local" / exp_name
    else:
        path = Path(output).expanduser()
        if not path.is_absolute():
            if not path.parts:
                path = root / "local" / exp_name
            elif path.parts[0] in FOLDER_TO_KIND:
                path = root / path
            elif len(path.parts) == 1:
                path = root / "local" / path

    if not path.suffix:
        path = path.with_suffix(".yml")
    return path.resolve()


def _with_section_comments(yaml_text: str, *, schema_line: str) -> str:
    out: list[str] = [schema_line]
    for line in yaml_text.splitlines():
        if line and not line.startswith((" ", "-")):
            section = line.split(":", 1)[0]
            title = SECTION_HEADERS.get(section)
            if title:
                if out[-1] != "":
                    out.append("")
                out.extend(_section_comment(title))
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _section_comment(title: str) -> list[str]:
    border = "# ========================================================"
    return [border, f"# {title}", border]


__all__ = [
    "KIND_TO_FOLDER",
    "TrainingConfigEntry",
    "TrainingConfigKind",
    "TrainingPresetName",
    "create_training_config_from_preset",
    "create_training_config_from_template",
    "discover_training_configs",
    "load_training_config",
    "resolve_training_config_path",
    "resolve_training_preset_path",
    "resolve_training_template_path",
    "save_training_config_yaml",
    "strip_config_metadata",
    "validate_training_config_dict",
    "validate_training_template_dict",
]
