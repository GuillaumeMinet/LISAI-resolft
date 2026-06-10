from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from lisai.runs.io import read_run_metadata

from ..models import ContinueTrainingConfig, ExperimentConfig, ResolvedExperiment, RetrainConfig
from .merge import deep_merge
from .metadata import split_config_metadata
from .yaml import load_yaml

if TYPE_CHECKING:
    from lisai.infra.paths import Paths


def _dget(d: dict, path: str, default=None):
    " Small helper to get nested dict values from dotted path."
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _dset(d: dict, path: str, value):
    """ Small helper to set nested dict value by dotted path.
    Example:
        _dset(cfg, "experiment.mode", "train") will ensure 
            {
            "experiment": {
                "mode": "train"
                }
            }
    """
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _normalize_mode(cfg: dict) -> str:
    """ Ensure mode is known, and that it is found in 
    cfg under {experiment: {"mode": mode}}"""
    
    mode = _dget(cfg, "experiment.mode", None) or cfg.get("mode") or "train"
    if mode == "resume":
        mode = "continue_training"
    if mode not in {"train", "continue_training", "retrain"}:
        raise ValueError(f"Unknown mode: {mode}")
    _dset(cfg, "experiment.mode", mode)
    cfg.pop("mode", None)
    return mode


def _authoring_model_for_mode(mode: str):
    if mode == "train":
        return ExperimentConfig
    if mode == "continue_training":
        return ContinueTrainingConfig
    if mode == "retrain":
        return RetrainConfig
    raise ValueError(f"Unknown mode: {mode}")


def _normalize_load_model(mode: str, cfg: dict, paths: Paths) -> dict:
    """Canonicalize ``cfg["load_model"]`` for config resolution.

    This does not load checkpoint weights. It converts user-authored
    ``load_model`` settings into the runtime shape expected downstream:
    ``enabled``, ``source``, ``run_dir``, and a ``checkpoint`` mapping with
    ``method``, ``selector``, ``epoch``, and ``filename``.

    Fresh ``train`` runs always get loading disabled. Continue/retrain modes
    accept both current and legacy checkpoint option names, resolve the source
    run directory from either canonical dataset/experiment fields or a direct
    path, write the normalized mapping back into ``cfg``, and return it.
    """
    if mode == "train":
        enabled = False
        raw = {}
    else:
        raw = cfg.get("load_model") or {}
        enabled = bool(raw.get("enabled", raw))

    checkpoint = raw.get("checkpoint") or {}

    # canonical runtime shape of load_model
    out = {
        "enabled": enabled,
        "source": raw.get("source"),
        "run_dir": raw.get("run_dir"),
        "checkpoint": {
            "method": raw.get("load_method", raw.get("method", checkpoint.get("method"))),
            "selector": raw.get("best_or_last", raw.get("selector", checkpoint.get("selector"))),
            "epoch": raw.get("epoch_number", raw.get("epoch", checkpoint.get("epoch"))),
            "filename": raw.get("model_name", raw.get("filename", checkpoint.get("filename"))),
        },
    }

    if not enabled:
        cfg["load_model"] = out
        return out

    # if run_dir already available (e.g. config already previously normalized)
    if out["run_dir"]:
        out["run_dir"] = str(Path(out["run_dir"]).resolve())
        if out["source"] is None:
            out["source"] = "path"
        cfg["load_model"] = out
        return out

    # canonical loading with dataset name, model subfolder and experiment name
    canonical = raw.get("canonical_load", True)
    if canonical:
        ds_name = raw.get("dataset_name", _dget(cfg, "data.dataset_name"))
        models_subfolder = raw.get("subfolder", _dget(cfg, "routing.models_subfolder", ""))
        exp_name = raw.get("exp_name") or raw.get("name")

        if not ds_name:
            raise ValueError("load_model: missing dataset_name (canonical_load=True)")
        if not exp_name:
            raise ValueError("load_model: missing exp_name (canonical_load=True)")

        run_dir = paths.run_dir(dataset_name=ds_name, models_subfolder=models_subfolder, exp_name=exp_name)
        out["source"] = "canonical"
        out["run_dir"] = str(run_dir.resolve())
    
    # loading with full path (= source run directory path)
    else:
        full_path = raw.get("model_full_path") or raw.get("run_dir") or raw.get("path")
        if not full_path:
            raise ValueError("load_model: missing model_full_path (canonical_load=False)")
        out["source"] = "path"
        out["run_dir"] = str(Path(full_path).resolve())

    cfg["load_model"] = out
    return out


def _load_origin_cfg(origin_run_dir: Path, paths: Paths) -> dict:
    origin_cfg_path = paths.cfg_train_path(run_dir=origin_run_dir)
    return load_yaml(origin_cfg_path)


def _merge_root_override(origin_cfg: dict, user_cfg: dict, root: str) -> None:
    if root not in user_cfg:
        return

    override = deepcopy(user_cfg[root])
    base = origin_cfg.get(root)
    if isinstance(base, dict) and isinstance(override, dict):
        origin_cfg[root] = deep_merge(base, override)
    else:
        origin_cfg[root] = override


def _apply_mode_resolution(user_cfg: dict, mode: str, paths: Paths) -> dict:
    """Buid effective config for retrain/continue mode.
    Loads the origin run config, then apply only the override roots allowed for that
    mode so the original run remains the source of truth.
    """

    if mode == "train": # safety caution
        return user_cfg

    origin_dir = Path(_dget(user_cfg, "experiment.origin_run_dir"))
    origin_cfg = _load_origin_cfg(origin_dir, paths)

    # logic: resume mode should only change minimal things, but retrain can have major changes
    override_roots_by_mode = {
        "continue_training": ["experiment", "training", "saving", "tensorboard", "load_model", "recovery"],
        "retrain": [
            "experiment",
            "routing",
            "data",
            "training",
            "normalization",
            "model_norm_prm",
            "loss_function",
            "noise_model",
            "saving",
            "tensorboard",
            "load_model",
        ],
    }

    for root in override_roots_by_mode[mode]:
        _merge_root_override(origin_cfg, user_cfg, root)

    _dset(origin_cfg, "experiment.origin_run_dir", str(origin_dir.resolve()))
    _dset(origin_cfg, "experiment.mode", mode)
    return origin_cfg


def _apply_safe_resume_resolution(mode: str, cfg: dict, paths: Paths) -> None:
    if mode != "continue_training":
        return

    recovery_cfg = (((cfg.get("recovery") or {}).get("hdn_safe_resume")) or {})
    if not recovery_cfg.get("enabled", True):
        return
    if not recovery_cfg.get("auto_use_safe_checkpoint_on_continue", True):
        return

    load_model = cfg.get("load_model") or {}
    if not load_model.get("enabled", False):
        return

    checkpoint = load_model.get("checkpoint") or {}

    # Respect explicit user checkpoint choices.
    explicit_filename = checkpoint.get("filename")
    explicit_epoch = checkpoint.get("epoch")
    explicit_selector = checkpoint.get("selector")
    if explicit_filename is not None or explicit_epoch is not None:
        return
    if explicit_selector not in (None, "last"):
        return

    origin_run_dir = _dget(cfg, "experiment.origin_run_dir") or load_model.get("run_dir")
    if not origin_run_dir:
        return

    origin_run_dir = Path(origin_run_dir).resolve()

    try:
        metadata = read_run_metadata(origin_run_dir)
    except Exception:
        return

    recovery_checkpoint_filename = getattr(metadata, "recovery_checkpoint_filename", None)
    if metadata.status != "failed" or not recovery_checkpoint_filename:
        return

    recovery_ckpt_path = paths.checkpoint_path(
        run_dir=origin_run_dir,
        model_name=recovery_checkpoint_filename,
    )
    if not recovery_ckpt_path.exists():
        return

    checkpoint["method"] = "state_dict"
    checkpoint["filename"] = recovery_checkpoint_filename
    checkpoint["selector"] = None
    checkpoint["epoch"] = None
    load_model["checkpoint"] = checkpoint
    cfg["load_model"] = load_model

def _resolve_loaded_config(
    exp_cfg: dict,
    *,
    stg = None,
) -> ResolvedExperiment:
    
    # init settings and lisai Paths, and get project and data config from settings
    if stg is None:
        from ..settings import settings as stg
    from lisai.infra.paths import Paths

    project_cfg = stg.project_cfg.model_dump(mode="python")
    data_cfg = stg.data_cfg.model_dump(mode="python")
    paths = Paths(stg)

    # save experiment config
    exp_cfg, metadata = split_config_metadata(exp_cfg)
    if metadata is not None and metadata.get("kind") == "template":
        raise ValueError("Training templates must be instantiated before they can be resolved or trained.")
    changeme_paths = _changeme_paths(exp_cfg)
    if changeme_paths:
        joined = ", ".join(changeme_paths)
        raise ValueError(f"Training config still contains CHANGEME placeholder(s): {joined}")
    exp_cfg = deepcopy(exp_cfg)
    
    # mode normalization and validation of experiment config
    mode = _normalize_mode(exp_cfg)
    _authoring_model_for_mode(mode).model_validate(exp_cfg)

    # train mode
    if mode == "train":
        # merge configs, normalize load model config and config final validation
        cfg = deep_merge(project_cfg, data_cfg)
        cfg = deep_merge(cfg, exp_cfg)
        _normalize_mode(cfg) # safety protection of mode
        _normalize_load_model(mode, cfg, paths)
        return ResolvedExperiment.model_validate(cfg)

    # other modes: continue/retrain

    # 1. merge exp config with project and data cfgs
    load_resolution_cfg = deep_merge(project_cfg, data_cfg)
    load_resolution_cfg = deep_merge(load_resolution_cfg, exp_cfg)
    _normalize_mode(load_resolution_cfg) # safety protection of mode

    
    # 2. normalize load model config, resolving run_dir
    load_model = _normalize_load_model(mode, load_resolution_cfg, paths)
    if not load_model.get("enabled", False):
        raise ValueError(f"Mode '{mode}' requires 'load_model' section.")

    # 3. save and normalize user cfg with correct mode and load_model config
    user_cfg = deepcopy(exp_cfg)
    _normalize_mode(user_cfg)
    user_cfg["load_model"] = load_model
    if mode == "continue_training" and "recovery" in load_resolution_cfg:
        # save recovery settings
        user_cfg["recovery"] = deepcopy(load_resolution_cfg["recovery"])
    _dset(user_cfg, "experiment.origin_run_dir", load_model["run_dir"])

    # 4. merge step: load original config, and merges allowed override
    cfg = _apply_mode_resolution(user_cfg, mode, paths)

    # 5. clean up and validate final config for safety
    _normalize_mode(cfg)
    _normalize_load_model(mode, cfg, paths)
    _apply_safe_resume_resolution(mode, cfg, paths)
    return ResolvedExperiment.model_validate(cfg)


def resolve_config(
    experiment_cfg_path: str | Path,
    *,
    stg=None,
) -> ResolvedExperiment:
    exp_cfg = load_yaml(Path(experiment_cfg_path))
    return _resolve_loaded_config(exp_cfg, stg=stg)


def resolve_config_dict(
    experiment_cfg: dict,
    *,
    stg=None,
) -> ResolvedExperiment:
    if not isinstance(experiment_cfg, dict):
        raise TypeError("experiment_cfg must be a dictionary")

    return _resolve_loaded_config(experiment_cfg, stg=stg)


def _changeme_paths(value, *, path: str = "") -> list[str]:
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

def prune_config_for_saving(cfg: ResolvedExperiment) -> dict:
    out = cfg.model_dump(exclude_none=True)

    mode = cfg.experiment.mode
    if mode == "train":
        out.pop("load_model", None)
        out.get("experiment", {}).pop("origin_run_dir", None)

    if not bool(cfg.saving.enabled):
        out.pop("saving", None)

    if not bool(cfg.tensorboard.enabled):
        out.pop("tensorboard", None)

    return out
