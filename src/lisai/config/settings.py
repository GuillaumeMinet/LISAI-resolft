from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import DataConfig, LocalConfig, ProjectConfig
from .models.inference import ResolvedInferenceConfig
from .models.inference.presets import POST_TRAINING_OVERRIDES
from .io.yaml import load_yaml, save_yaml


_LOCAL_CONFIG_SCHEMA_HINT = "# yaml-language-server: $schema=./schema/local-config.schema.json"
_INFERENCE_DEFAULTS_SCHEMA_HINT = "# yaml-language-server: $schema=../../schema/inference-defaults.schema.json"
_INFERENCE_OVERRIDES_SCHEMA_HINT = "# yaml-language-server: $schema=../../schema/inference.schema.json"


class AttrDict(dict):
    """dict with attribute access, so '{paths.roots.data_dir}' works with str.format."""
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any):
        self[key] = value


def _to_attr(x: Any) -> Any:
    """
    Recursively convert dictionaries into AttrDict to allow attribute-style
    access required by ``str.format`` templates (e.g. ``{paths.roots.data_dir}``).

    - dict  → AttrDict with recursive conversion
    - list  → elements recursively converted
    - other → returned unchanged

    Used to build the formatting context for project path templates.
    """
    if isinstance(x, dict):
        ad = AttrDict()
        for k, v in x.items():
            ad[k] = _to_attr(v)
        return ad
    if isinstance(x, list):
        return [_to_attr(v) for v in x]
    return x


def _legacy_checkout_configs_root() -> Path:
    """Find the existing checkout configuration only at a default-loading boundary.

    A later workspace milestone will replace this legacy location selector.
    """
    return Path(__file__).resolve().parents[3] / "configs"


def load_default_settings() -> Settings:
    """Load the legacy checkout configuration afresh for a CLI/API operation.

    This function never initializes missing files. Library calls must not prompt
    for paths or modify an installation implicitly.
    """
    return Settings.from_config_dir(_legacy_checkout_configs_root())


def _migrate_local_inference_output(raw: dict) -> bool:
    """Migrate the deprecated flat output_mode during *explicit* setup only."""
    inference = raw.get("inference")
    if not isinstance(inference, dict) or "output_mode" not in inference:
        return False
    if "output" in inference:
        raise ValueError(
            "local inference config cannot define both output_mode and output."
        )

    mode = inference.pop("output_mode")
    inference["output"] = {
        "mode": mode,
        "save_input_mode": "if_not_in_place",
    }
    return True


def _ensure_schema_hint(path: Path, hint: str) -> None:
    text = path.read_text(encoding="utf-8")
    if hint not in text.splitlines():
        path.write_text(f"{hint}\n{text}", encoding="utf-8")


def _ensure_local_inference_configs(configs_root: Path) -> None:
    """Install missing legacy inference defaults during explicit setup only."""
    local_dir = configs_root / "inference" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    defaults_path = local_dir / "defaults.yml"
    if not defaults_path.exists():
        save_yaml(ResolvedInferenceConfig().model_dump(mode="json"), defaults_path)
    _ensure_schema_hint(defaults_path, _INFERENCE_DEFAULTS_SCHEMA_HINT)

    post_training_path = local_dir / "post_training.yml"
    if not post_training_path.exists():
        save_yaml(POST_TRAINING_OVERRIDES, post_training_path)
    _ensure_schema_hint(post_training_path, _INFERENCE_OVERRIDES_SCHEMA_HINT)


def prepare_legacy_config_dir(
    configs_root: str | Path,
    *,
    data_root: str | Path | None = None,
    interactive: bool = False,
) -> dict:
    """Explicitly initialize/migrate legacy files, never called by ``Settings``.

    Intended for a CLI setup/preflight command. An absent local config requires
    an explicit data root or ``interactive=True``; library imports never prompt.
    """
    configs_root = Path(configs_root).expanduser().resolve()
    local_path = configs_root / "local_config.yml"

    if local_path.is_file():
        raw = load_yaml(local_path)
        if _migrate_local_inference_output(raw):
            save_yaml(raw, local_path)
    else:
        if data_root is None:
            if not interactive:
                raise FileNotFoundError(
                    f"Missing LISAI local config: {local_path}. "
                    "Run explicit setup or supply a data_root."
                )
            default_root = "E:/dl_monalisa" if os.name == "nt" else "/data/dl_monalisa"
            entered = input(f"Enter absolute path to Data Root [default: {default_root}]: ").strip()
            data_root = entered or default_root
        raw = {
            "infrastructure": {"data_root": str(Path(data_root).expanduser().resolve())},
            "inference": {
                "output": {"mode": "default", "save_input_mode": "if_not_in_place"},
                "inference_dir": "default",
            },
            "console": {"progress_bar": None},
        }
        local_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(raw, local_path)

    _ensure_schema_hint(local_path, _LOCAL_CONFIG_SCHEMA_HINT)
    _ensure_local_inference_configs(configs_root)
    return raw


class Settings:
    """Validated, read-only view of one explicitly selected legacy config tree.

    Construction never creates or modifies files and never prompts. Low-level
    template expansion remains owned here; ``Paths`` builds domain paths from it.
    """

    def __init__(self, configs_root: str | Path):
        self.CONFIGS_ROOT = Path(configs_root).expanduser().resolve()
        self.PROJECT_ROOT = self.CONFIGS_ROOT.parent
        self._local_yaml_path = self.CONFIGS_ROOT / "local_config.yml"
        self._project_yaml_path = self.CONFIGS_ROOT / "project_config.yml"
        self._data_yaml_path = self.CONFIGS_ROOT / "data_config.yml"

        self.local_cfg: LocalConfig = LocalConfig.model_validate(
            self._load_required(self._local_yaml_path)
        )
        self.project_cfg: ProjectConfig = ProjectConfig.model_validate(
            self._load_required(self._project_yaml_path)
        )
        self.data_cfg: DataConfig = DataConfig.model_validate(
            self._load_required(self._data_yaml_path)
        )
        self._ctx = self._build_context()

    @classmethod
    def from_config_dir(cls, configs_root: str | Path) -> Settings:
        """Load the three existing YAML files from an explicit config directory."""
        return cls(configs_root)

    @staticmethod
    def _load_required(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Critical LISAI config missing: {path}")
        return load_yaml(path)

    def _build_context(self) -> AttrDict:
        data_root = Path(self.local_cfg.infrastructure.data_root).resolve()
        code_dir = self.PROJECT_ROOT.resolve()

        ctx = AttrDict(
            data_root=str(data_root),
            code_dir=str(code_dir),
            paths=AttrDict(
                roots=AttrDict(),
                templates=AttrDict(),
            ),
        )

        # Provide code_dir for templates
        ctx.paths.roots.code_dir = str(code_dir)

        # Resolve roots (depend only on local infrastructure plus explicit local overrides).
        for key, tmpl in (self.project_cfg.paths.roots or {}).items():
            if key == "run_container_dirname":
                text = str(tmpl).strip().strip("/\\")
                if not text:
                    raise ValueError("project.paths.roots.run_container_dirname must not be empty.")
                ctx.paths.roots[key] = text
                continue

            if key == "inference_dir" and self.local_cfg.inference.inference_dir != "default":
                value = self.local_cfg.inference.inference_dir
            else:
                value = tmpl.format(**ctx)
            value = str(Path(os.path.normpath(value)).expanduser().resolve())
            ctx.paths.roots[key] = value

        # Store templates as-is (experiment-dependent keys can't be resolved yet)
        for key, tmpl in (self.project_cfg.paths.templates or {}).items():
            ctx.paths.templates[key] = tmpl

        return ctx


    # ==========================================
    # PUBLIC API
    # ==========================================
    
    @property
    def NAMING(self):
        # expose naming conventions
        return self.project_cfg.naming
    
    @property
    def PROJECT_CONFIG_PATH(self) -> Path:
        return self._project_yaml_path

    @property
    def DATA_ROOT(self) -> Path:
        return Path(self._ctx.data_root).resolve()

    @property
    def DATA_CONFIG_PATH(self) -> Path:
        return self._data_yaml_path
        
    @property
    def TRAINING_CONFIG_DIR(self):
        return self.CONFIGS_ROOT / "training"
    
    @property
    def INFERENCE_CONFIG_DIR(self):
        return self.CONFIGS_ROOT / "inference"
    
    @property
    def INFERENCE_OUTPUT_MODE(self) -> str:
        return self.local_cfg.inference.output.mode

    @property
    def INFERENCE_SAVE_INPUT_MODE(self) -> str:
        return self.local_cfg.inference.output.save_input_mode

    @property
    def LOCAL_INFERENCE_DIR(self) -> str:
        return self.local_cfg.inference.inference_dir

    @property
    def LOCAL_PROGRESS_BAR(self) -> bool | None:
        return self.local_cfg.console.progress_bar

    @property
    def INFERENCE_DEFAULT_CONFIG_NAME(self):
        return "local/defaults"
    
    @property
    def PREPROCESS_CONFIG_DIR(self):
        return self.CONFIGS_ROOT / "preprocess"
    
    @property
    def CONFIG_SUFFIXES(self):
        return (".yml", ".yaml")
    
    def resolve_path(self, template: str, **kwargs) -> Path:
        merged = AttrDict(self._ctx)
        for k, v in kwargs.items():
            merged[k] = _to_attr(v)
        s = template.format(**merged)
        return Path(os.path.normpath(s))

    def get_template_path(self, key: str, **kwargs) -> Path:
        tmpl = self._ctx.paths.templates.get(key)
        if tmpl is None:
            raise KeyError(f"Template key '{key}' not found in project.paths.templates")
        return self.resolve_path(tmpl, **kwargs)

    def get_data_filename(self, fmt: str, data_type: str, **kwargs) -> Path:
        fmt_cfg = self.data_cfg.format.get(fmt)
        if not fmt_cfg:
            raise ValueError(f"Unknown format: {fmt}")
        template = fmt_cfg.get(data_type)
        if not template:
            raise ValueError(f"Unknown type '{data_type}' for format '{fmt}'")

        filename = self.resolve_path(template, **kwargs)
        return filename
