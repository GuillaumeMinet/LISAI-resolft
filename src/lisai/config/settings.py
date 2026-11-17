from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import DataConfig, LocalConfig, LegacyLocalConfig, DeveloperLocalConfig, ProjectConfig
from .models.inference import ResolvedInferenceConfig
from .models.inference.presets import POST_TRAINING_OVERRIDES
from .io.yaml import load_yaml, save_yaml
from pydantic import ConfigDict, Field
from .models.local_config import LocalConsoleConfig, LocalInferenceConfig


_LOCAL_CONFIG_SCHEMA_HINT = "# yaml-language-server: $schema=./schema/legacy-local-config.schema.json"
_INFERENCE_DEFAULTS_SCHEMA_HINT = "# yaml-language-server: $schema=../../schema/inference-defaults.schema.json"
CURRENT_DEFAULTS_REVISION = 1

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
    """Legacy checkout discovery retained for explicit compatibility tooling only."""
    return Path(__file__).resolve().parents[3] / "configs"


def load_default_settings() -> Settings:
    """Python API default: workspace selection, never CLI developer activation.

    Only LISAI_HOME is considered; developer selection is CLI-only. No reads
    create or modify files, and there is no implicit home-directory workspace.
    """
    selected = os.environ.get("LISAI_HOME")
    if not selected:
        raise ValueError(
            "LISAI_HOME is not set. Set it to an initialized workspace, "
            "or pass Settings.from_workspace(path) explicitly."
        )
    workspace = Path(selected).expanduser()
    if not workspace.is_absolute():
        raise ValueError("LISAI_HOME must be an absolute workspace path")
    return Settings.from_workspace(workspace)


class WorkspaceFileConfig(ProjectConfig):
    """Strict workspace.yml adapter, reusing the existing ProjectConfig model."""

    model_config = ConfigDict(extra="forbid")
    defaults_revision: int = Field(default=0, ge=0)
    inference: LocalInferenceConfig = Field(default_factory=LocalInferenceConfig)
    console: LocalConsoleConfig = Field(default_factory=LocalConsoleConfig)


def _legacy_project_templates(raw: dict) -> dict:
    """One-time legacy read adapter; active templates never expose data_root."""
    paths = raw.get("paths", {})
    for section in ("roots", "templates"):
        values = paths.get(section, {})
        for key, value in values.items():
            if isinstance(value, str):
                value = value.replace("{data_root}", "{workspace_root}")
                value = value.replace("{paths.roots.code_dir}/tensorboard", "{workspace_root}/tensorboard")
                values[key] = value
    return raw



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
    """One immutable-on-load, explicitly selected configuration context.

    Workspace YAMLs are authoritative and the writable experiment root is
    independent. The legacy three-file loader is retained solely as an explicit
    migration/testing entry point; its old placeholders are translated on read.
    Construction never creates, rewrites or prompts for configuration.
    """

    def __init__(self, configs_root: str | Path):
        # Explicit legacy loading; no code should call this as a new-runtime default.
        self.CONFIGS_ROOT = Path(configs_root).expanduser().resolve()
        self.PROJECT_ROOT = self.CONFIGS_ROOT.parent  # legacy loader compatibility only
        self._local_yaml_path = self.CONFIGS_ROOT / "local_config.yml"
        self._project_yaml_path = self.CONFIGS_ROOT / "project_config.yml"
        self._data_yaml_path = self.CONFIGS_ROOT / "data_config.yml"
        legacy = LegacyLocalConfig.model_validate(self._load_required(self._local_yaml_path))
        self.WORKSPACE_ROOT = Path(legacy.infrastructure.data_root).expanduser().resolve()
        self.DEFAULTS_REVISION = None  # Legacy settings carry no workspace revision.
        self.local_cfg = LocalConfig.model_validate(
            legacy.model_dump(exclude={"infrastructure"})
        )
        self.project_cfg = ProjectConfig.model_validate(
            _legacy_project_templates(self._load_required(self._project_yaml_path))
        )
        self.data_cfg = DataConfig.model_validate(self._load_required(self._data_yaml_path))
        self._ctx = self._build_context()

    @classmethod
    def from_config_dir(cls, configs_root: str | Path) -> Settings:
        """Explicit read-only adapter for old three-file configuration trees."""
        return cls(configs_root)

    @classmethod
    def from_workspace(
        cls, workspace: str | Path, *,
        config_root: str | Path | None = None,
        local_config: str | Path | None = None,
    ) -> Settings:
        """Load workspace.yml + data.yml from workspace, user YAMLs from config_root.

        Only an explicitly provided local_config is loaded for optional
        developer preferences. Neither workspace nor config_root is searched
        for local.yml, and this constructor never performs CLI selection.
        """
        instance = cls.__new__(cls)
        instance.WORKSPACE_ROOT = Path(workspace).expanduser().resolve()
        owned = instance.WORKSPACE_ROOT / "configs"
        instance.CONFIGS_ROOT = Path(config_root).expanduser().resolve() if config_root is not None else owned
        instance._project_yaml_path = owned / "workspace.yml"
        instance._data_yaml_path = owned / "data.yml"
        instance._local_yaml_path = None
        if local_config is not None:
            instance._local_yaml_path = Path(local_config).expanduser().resolve()

        source = instance._load_required(instance._project_yaml_path)
        workspace_file = WorkspaceFileConfig.model_validate(source)
        instance.DEFAULTS_REVISION = workspace_file.defaults_revision
        # Pydantic strict ProjectConfig remains the public project_cfg.
        instance.project_cfg = ProjectConfig.model_validate(
            workspace_file.model_dump(exclude={"inference", "console", "defaults_revision"})
        )
        instance.data_cfg = DataConfig.model_validate(instance._load_required(instance._data_yaml_path))
        effective = {
            "inference": workspace_file.inference.model_dump(),
            "console": workspace_file.console.model_dump(),
        }
        if instance._local_yaml_path is not None:
            external = DeveloperLocalConfig.model_validate(
                instance._load_required(instance._local_yaml_path)
            )
            # Whitelist: only explicitly authored console/inference values.
            # This is not a general-purpose deep merge.
            authored = external.model_dump(exclude_unset=True)
            for field, value in authored.get("console", {}).items():
                effective["console"][field] = value
            for field, value in authored.get("inference", {}).items():
                if field == "output":
                    effective["inference"]["output"].update(value)
                else:
                    effective["inference"][field] = value
        instance.local_cfg = LocalConfig.model_validate(effective)
        instance._ctx = instance._build_context()
        return instance

    @staticmethod
    def _load_required(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(
                f"Critical LISAI config missing: {path}. "
                f"Initialize its workspace explicitly with 'lisai workspace init {path.parent.parent}'."
            )
        return load_yaml(path)

    def _build_context(self) -> AttrDict:
        ctx = AttrDict(
            workspace_root=str(self.WORKSPACE_ROOT),
            paths=AttrDict(roots=AttrDict(), templates=AttrDict()),
        )
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
            ctx.paths.roots[key] = str(Path(os.path.normpath(value)).expanduser().resolve())
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
    def WORKSPACE_CONFIG_PATH(self) -> Path:
        return self._project_yaml_path

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
        return "defaults"
    
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
