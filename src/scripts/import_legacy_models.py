"""Import one legacy LISAI model folder as a current discoverable run.

Edit the fields in "USER SETTINGS" below, then run this file directly:

    python src/scripts/import_legacy_models.py

The script validates everything before writing:
- source config and state-dict checkpoints
- target canonical run folder does not already exist
- referenced HDN/LVAE noise model is available in the current LISAI noise-model root
- translated config validates as a current ResolvedExperiment
- legacy checkpoint weights can be loaded into the translated current model
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# Make the script runnable from a source checkout without requiring installation.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from lisai.config import prune_config_for_saving, save_yaml, settings
from lisai.config.models import ResolvedExperiment
from lisai.infra.fs import ensure_folder
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.infra.paths import Paths, model_filename
from lisai.infra.paths.model_subfolder import (
    group_path_from_model_subfolder,
    normalize_model_subfolder,
)
from lisai.models.params import LVAEParams, UNetParams, UNetRCANParams
from lisai.runs.identifiers import generate_run_id
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.lifecycle import stored_run_path
from lisai.runs.plotting import save_loss_plot_for_run
from lisai.runs.schema import RunMetadata, RunProvenance, TrainingSignature, utc_now
from lisai.runs.signature import (
    build_training_signature_from_resolved_config,
    count_trainable_parameters,
)

# ==============================
# USER SETTINGS
# ==============================

# Provenance
LEGACY_MODEL_PATH = r"E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_refinement\SNRavg\Avg_unpaired_Mltpl075notRdm_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005"

# Target canonical run location. TARGET_RUN_NAME is the final run folder name.
TARGET_DATASET = "vim_fixed_multi_snr"
TARGET_MODEL_SUBFOLDER = "Upsamp"
TARGET_RUN_NAME = "SnrHigh_unpaired_S075"

# Start with a dry run; set to False once the preflight output looks right.
DRY_RUN = False

# Keep a small audit copy of the old config under target/legacy_origin.
# Legacy loss/log files are copied to the canonical run root when present.
COPY_SMALL_LEGACY_ARTIFACTS = True


@dataclass(frozen=True)
class LegacyImportJob:
    legacy_model_path: Path
    target_dataset: str
    target_model_subfolder: str
    target_run_name: str
    dry_run: bool = True
    copy_small_legacy_artifacts: bool = True


@dataclass(frozen=True)
class LegacyModelSpec:
    architecture: str
    parameters: Any
    runtime_img_shape: tuple[int, int] | None
    model_norm_prm: dict[str, Any] | None
    noise_model_name: str | None


@dataclass(frozen=True)
class SelectedCheckpoint:
    selector: str
    source_path: Path
    epoch: int | None


@dataclass(frozen=True)
class ConvertedCheckpoint:
    selector: str
    source_path: Path
    epoch: int | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class ImportPlan:
    job: LegacyImportJob
    target_run_dir: Path
    resolved_cfg: ResolvedExperiment
    legacy_cfg: dict[str, Any]
    model_spec: LegacyModelSpec
    checkpoints: tuple[ConvertedCheckpoint, ...]
    best_val_loss: float | None
    metadata: RunMetadata
    trainable_params: int | None


def main() -> int:
    job = LegacyImportJob(
        legacy_model_path=Path(LEGACY_MODEL_PATH),
        target_dataset=TARGET_DATASET.strip(),
        target_model_subfolder=TARGET_MODEL_SUBFOLDER.strip(),
        target_run_name=TARGET_RUN_NAME.strip(),
        dry_run=bool(DRY_RUN),
        copy_small_legacy_artifacts=bool(COPY_SMALL_LEGACY_ARTIFACTS),
    )

    try:
        plan = build_import_plan(job)
    except Exception as exc:
        print(f"Preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print_import_plan(plan)
    if job.dry_run:
        print("\nDRY_RUN=True: no files were written.")
        return 0

    execute_import(plan)
    print(f"\nImported legacy run into: {plan.target_run_dir}")
    return 0


def build_import_plan(job: LegacyImportJob) -> ImportPlan:
    job = _normalize_job_settings(job)
    _validate_job_settings(job)

    paths = Paths(settings)
    legacy_cfg = _load_legacy_config(job.legacy_model_path)
    target_run_dir = paths.run_dir(
        dataset_name=job.target_dataset,
        models_subfolder=job.target_model_subfolder,
        exp_name=job.target_run_name,
    )

    if target_run_dir.exists():
        raise FileExistsError(f"Target run folder already exists: {target_run_dir}")

    current_cfg = translate_legacy_config(
        legacy_cfg,
        target_dataset=job.target_dataset,
        target_model_subfolder=job.target_model_subfolder,
        target_run_name=job.target_run_name,
    )
    resolved_cfg = ResolvedExperiment.model_validate(current_cfg)
    model_spec = build_model_spec(legacy_cfg)
    _verify_noise_model_available(model_spec, paths)

    selected_checkpoints = select_legacy_checkpoints(job.legacy_model_path)
    converted: list[ConvertedCheckpoint] = []
    trainable_params = None
    for selected in selected_checkpoints:
        converted_checkpoint, model = convert_checkpoint(selected, model_spec, paths)
        converted.append(converted_checkpoint)
        if trainable_params is None:
            trainable_params = count_trainable_parameters(model)

    best_val_loss = _best_val_loss(job.legacy_model_path / "loss.txt")
    metadata = build_import_metadata(
        job=job,
        target_run_dir=target_run_dir,
        resolved_cfg=resolved_cfg,
        selected_checkpoints=tuple(converted),
        best_val_loss=best_val_loss,
        trainable_params=trainable_params,
    )

    return ImportPlan(
        job=job,
        target_run_dir=target_run_dir,
        resolved_cfg=resolved_cfg,
        legacy_cfg=legacy_cfg,
        model_spec=model_spec,
        checkpoints=tuple(converted),
        best_val_loss=best_val_loss,
        metadata=metadata,
        trainable_params=trainable_params,
    )


def _validate_job_settings(job: LegacyImportJob):
    missing: list[str] = []
    if not str(job.legacy_model_path).strip() or str(job.legacy_model_path) == ".":
        missing.append("LEGACY_MODEL_PATH")
    if not job.target_dataset:
        missing.append("TARGET_DATASET")
    if not job.target_model_subfolder:
        missing.append("TARGET_MODEL_SUBFOLDER")
    if not job.target_run_name:
        missing.append("TARGET_RUN_NAME")
    if missing:
        raise ValueError("Fill the script USER SETTINGS first: " + ", ".join(missing))

    if not job.legacy_model_path.is_dir():
        raise FileNotFoundError(f"Legacy model folder not found: {job.legacy_model_path}")
    if not (job.legacy_model_path / "config_train.json").is_file():
        raise FileNotFoundError(
            f"Legacy config not found: {job.legacy_model_path / 'config_train.json'}"
        )


def _normalize_job_settings(job: LegacyImportJob) -> LegacyImportJob:
    return LegacyImportJob(
        legacy_model_path=job.legacy_model_path,
        target_dataset=_normalize_required_folder_name(
            job.target_dataset,
            field_name="TARGET_DATASET",
        ),
        target_model_subfolder=_normalize_target_model_subfolder(job.target_model_subfolder),
        target_run_name=_normalize_required_folder_name(
            job.target_run_name,
            field_name="TARGET_RUN_NAME",
        ),
        dry_run=job.dry_run,
        copy_small_legacy_artifacts=job.copy_small_legacy_artifacts,
    )


def _normalize_required_folder_name(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must be a folder name, not a path: {value!r}")
    return text


def _normalize_target_model_subfolder(value: str) -> str:
    normalized = normalize_model_subfolder(value)
    if normalized is None:
        return ""

    first_part = normalized.split("/", 1)[0].lower()
    current_run_container = Paths(settings).run_container_dirname().lower()
    stale_containers = {"models", current_run_container}
    if first_part in stale_containers:
        raise ValueError(
            "TARGET_MODEL_SUBFOLDER must be relative inside the current run container. "
            f"Remove the leading {first_part!r} path component from {value!r}."
        )
    return normalized


def _load_legacy_config(legacy_model_path: Path) -> dict[str, Any]:
    cfg_path = legacy_model_path / "config_train.json"
    with cfg_path.open("r", encoding="utf-8-sig") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Legacy config must be a JSON object: {cfg_path}")
    return cfg


def translate_legacy_config(
    legacy_cfg: dict[str, Any],
    *,
    target_dataset: str,
    target_model_subfolder: str,
    target_run_name: str,
) -> dict[str, Any]:
    target_dataset = _normalize_required_folder_name(
        target_dataset,
        field_name="target_dataset",
    )
    target_model_subfolder = _normalize_target_model_subfolder(target_model_subfolder)
    target_run_name = _normalize_required_folder_name(
        target_run_name,
        field_name="target_run_name",
    )
    data_cfg, data_subfolder = _translate_legacy_data_cfg(
        legacy_cfg,
        target_dataset=target_dataset,
    )
    training_cfg = dict(legacy_cfg.get("training_prm") or {})
    if "learning_rate" not in training_cfg and "lr" in training_cfg:
        training_cfg["learning_rate"] = training_cfg["lr"]
    if "progress_bar" not in training_cfg and "pbar" in training_cfg:
        training_cfg["progress_bar"] = training_cfg["pbar"]

    saving_cfg = dict(legacy_cfg.get("saving_prm") or {})
    model_section = _legacy_model_section(legacy_cfg)
    current = {
        "experiment": {
            "mode": "train",
            "exp_name": target_run_name,
            "overwrite": False,
            "post_training_inference": False,
        },
        "routing": {
            "data_subfolder": data_subfolder,
            "models_subfolder": target_model_subfolder,
            "inference_subfolder": target_model_subfolder,
        },
        "data": data_cfg,
        "model": model_section,
        "training": training_cfg,
        "normalization": dict(legacy_cfg.get("normalization") or {}),
        "model_norm_prm": legacy_cfg.get("model_norm_prm"),
        "noise_model": legacy_cfg.get("noise_model"),
        "saving": {
            "enabled": True,
            "canonical_save": True,
            "state_dict": True,
            "entire_model": False,
            "overwrite_best": bool(saving_cfg.get("overwrite_best", True)),
            "validation_images": bool(saving_cfg.get("save_validation_images", False)),
            "validation_freq": int(saving_cfg.get("save_validation_freq", 10) or 10),
        },
        "tensorboard": {"enabled": False},
    }
    inference_cfg = _legacy_inference_defaults(model_section)
    if inference_cfg:
        current["inference"] = inference_cfg
    return current


def _translate_legacy_data_cfg(
    legacy_cfg: dict[str, Any],
    *,
    target_dataset: str,
) -> tuple[dict[str, Any], str]:
    data_cfg = _legacy_data_cfg(legacy_cfg)
    data_subfolder = _normalize_relative_subpath(
        data_cfg.pop("subfolder", ""),
        field_name="data_prm.subfolder",
    )

    if "input" not in data_cfg and data_cfg.get("inp") is not None:
        data_cfg["input"] = data_cfg["inp"]
    if "target" not in data_cfg and data_cfg.get("gt") is not None:
        data_cfg["target"] = data_cfg["gt"]

    # These were legacy path/runtime fields. Current LISAI resolves dataset
    # paths from routing.data_subfolder and the configured training data root.
    legacy_runtime_keys = (
        "full_data_path",
        "data_dir",
        "subfolder",
        "inp",
        "gt",
        "patch_info",
        "volumetric",
    )
    for key in legacy_runtime_keys:
        data_cfg.pop(key, None)

    data_cfg["dataset_name"] = target_dataset
    data_cfg["canonical_load"] = True
    return data_cfg, data_subfolder


def _legacy_data_cfg(legacy_cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(legacy_cfg.get("data_prm") or {})
    if not data_cfg:
        raise ValueError("Only legacy configs with `data_prm` are supported by this importer.")
    return data_cfg


def _normalize_relative_subpath(value: Any, *, field_name: str) -> str:
    if value in (None, ""):
        return ""
    text = str(value).replace("\\", "/").strip().strip("/")
    if not text:
        return ""
    if re.match(r"^[A-Za-z]:", text) or text.startswith("/"):
        raise ValueError(f"`{field_name}` must be relative, not an absolute path: {value!r}")
    normalized = PurePosixPath(text).as_posix()
    if normalized in {"", "."}:
        return ""
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"`{field_name}` must not escape the dataset root: {value!r}")
    return normalized


def _legacy_model_section(legacy_cfg: dict[str, Any]) -> dict[str, Any]:
    architecture = legacy_cfg.get("model_architecture")
    if architecture is None and legacy_cfg.get("is_lvae"):
        architecture = "lvae"
    if architecture is None:
        raise ValueError(
            "Only legacy configs with `model_architecture` or `is_lvae` are supported."
        )

    architecture = str(architecture).strip().lower()
    if architecture == "unetrcan":
        architecture = "unet_rcan"

    params = _translate_legacy_model_params(architecture, dict(legacy_cfg.get("model_prm") or {}))
    return {"architecture": architecture, "parameters": params}


def _translate_legacy_model_params(architecture: str, params: dict[str, Any]) -> dict[str, Any]:
    if architecture == "lvae":
        out = dict(params)
        out.pop("img_shape", None)
        out.pop("norm_prm", None)
        if "batchnorm" in out and "norm" not in out:
            out["norm"] = "batch" if out["batchnorm"] else None
        LVAEParams.model_validate(out)
        return out

    if architecture == "unet":
        out = dict(params)
        out.setdefault("upsampling_factor", 1)
        out.setdefault("remove_skip_con", 1)
        if isinstance(out.get("remove_skip_con"), bool):
            out["remove_skip_con"] = int(out["remove_skip_con"])
        UNetParams.model_validate(out)
        return out

    if architecture == "unet_rcan":
        out = dict(params)
        out.setdefault("upsampling_factor", 1)
        unet_prm = dict(out.get("UNet_prm") or {})
        unet_prm.setdefault("remove_skip_con", 1)
        if isinstance(unet_prm.get("remove_skip_con"), bool):
            unet_prm["remove_skip_con"] = int(unet_prm["remove_skip_con"])
        out["UNet_prm"] = unet_prm

        rcan_prm = dict(out.get("RCAN_prm") or {})
        legacy_upsamp = rcan_prm.get("upsamp")
        if legacy_upsamp and "upsamp_kernel_factor" not in rcan_prm:
            rcan_prm["upsamp_kernel_factor"] = 2
        rcan_prm.pop("upsamp", None)
        out["RCAN_prm"] = rcan_prm
        out.setdefault("upsampling_net", "rcan" if legacy_upsamp else "unet")
        UNetRCANParams.model_validate(out)
        return out

    raise ValueError(f"Unsupported legacy architecture: {architecture!r}")


def _legacy_inference_defaults(model_section: dict[str, Any]) -> dict[str, Any]:
    """Return saved inference defaults for imported legacy models."""
    architecture = model_section["architecture"]
    if architecture != "unet_rcan":
        return {}

    params = UNetRCANParams.model_validate(model_section["parameters"])
    if params.effective_upsampling_factor() == 1:
        return {"default_tiling_size": 2000}
    return {}


def build_model_spec(legacy_cfg: dict[str, Any]) -> LegacyModelSpec:
    model_section = _legacy_model_section(legacy_cfg)
    architecture = model_section["architecture"]
    parameters = model_section["parameters"]
    model_prm = dict(legacy_cfg.get("model_prm") or {})

    if architecture == "lvae":
        model_params = LVAEParams.model_validate(parameters)
        runtime_img_shape = _runtime_img_shape(model_prm, legacy_cfg)
        model_norm_prm = dict(
            legacy_cfg.get("model_norm_prm")
            or model_prm.get("norm_prm")
            or {"data_mean": 0, "data_std": 1, "data_mean_gt": None, "data_std_gt": None}
        )
    elif architecture == "unet":
        model_params = UNetParams.model_validate(parameters)
        runtime_img_shape = None
        model_norm_prm = legacy_cfg.get("model_norm_prm")
    elif architecture == "unet_rcan":
        model_params = UNetRCANParams.model_validate(parameters)
        runtime_img_shape = None
        model_norm_prm = legacy_cfg.get("model_norm_prm")
    else:
        raise ValueError(f"Unsupported legacy architecture: {architecture!r}")

    return LegacyModelSpec(
        architecture=architecture,
        parameters=model_params,
        runtime_img_shape=runtime_img_shape,
        model_norm_prm=model_norm_prm,
        noise_model_name=legacy_cfg.get("noise_model"),
    )


def _runtime_img_shape(model_prm: dict[str, Any], legacy_cfg: dict[str, Any]) -> tuple[int, int]:
    raw = model_prm.get("img_shape")
    if raw is None:
        patch_size = (legacy_cfg.get("data_prm") or {}).get("patch_size", 64)
        raw = [patch_size, patch_size]
    if isinstance(raw, int):
        return int(raw), int(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    raise ValueError(f"Invalid legacy LVAE img_shape: {raw!r}")


def _verify_noise_model_available(model_spec: LegacyModelSpec, paths: Paths):
    if model_spec.architecture != "lvae":
        return
    if not model_spec.noise_model_name:
        raise ValueError("Legacy LVAE/HDN config is missing `noise_model`.")
    nm_path = paths.noise_model_path(noiseModel_name=model_spec.noise_model_name)
    norm_path = paths.noise_model_norm_prm_path(noiseModel_name=model_spec.noise_model_name)
    missing = [path for path in (nm_path, norm_path) if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Canonical noise model files are missing. Copy the noise model into "
            f"the current LISAI noise-model root first:\n{joined}"
        )


def select_legacy_checkpoints(legacy_model_path: Path) -> tuple[SelectedCheckpoint, ...]:
    best = _select_best_checkpoint(legacy_model_path)
    last = _select_last_checkpoint(legacy_model_path)
    if best is None and last is None:
        raise FileNotFoundError(f"No legacy state-dict checkpoint found in {legacy_model_path}")
    if best is None:
        assert last is not None
        best = SelectedCheckpoint("best", last.source_path, last.epoch)
    if last is None:
        assert best is not None
        last = SelectedCheckpoint("last", best.source_path, best.epoch)
    return best, last


def _select_best_checkpoint(legacy_model_path: Path) -> SelectedCheckpoint | None:
    direct = legacy_model_path / "model_best_state_dict.pt"
    if direct.exists():
        return SelectedCheckpoint("best", direct, _checkpoint_epoch(direct))

    best_epoch = _best_epoch_from_loss(legacy_model_path / "loss.txt")
    if best_epoch is not None:
        epoch_path = legacy_model_path / f"model_epoch_{best_epoch}_state_dict.pt"
        if epoch_path.exists():
            return SelectedCheckpoint("best", epoch_path, best_epoch)
    return None


def _select_last_checkpoint(legacy_model_path: Path) -> SelectedCheckpoint | None:
    direct = legacy_model_path / "model_last_state_dict.pt"
    if direct.exists():
        return SelectedCheckpoint("last", direct, _checkpoint_epoch(direct))

    epoch_paths = sorted(
        legacy_model_path.glob("model_epoch_*_state_dict.pt"),
        key=lambda path: (_checkpoint_epoch(path) is None, _checkpoint_epoch(path) or -1),
    )
    if epoch_paths:
        path = epoch_paths[-1]
        return SelectedCheckpoint("last", path, _checkpoint_epoch(path))
    return None


def convert_checkpoint(
    selected: SelectedCheckpoint,
    model_spec: LegacyModelSpec,
    paths: Paths,
) -> tuple[ConvertedCheckpoint, Any]:
    model = _instantiate_model(model_spec, paths)
    loaded = _torch_load(selected.source_path)
    state_dict = _extract_model_state_dict(loaded, selected.source_path)

    missing, unexpected, shape_mismatches = _state_dict_diff(model, state_dict)
    unsafe_missing = [
        key for key in missing if not _is_allowed_inactive_missing_key(key, model_spec)
    ]
    if unexpected or unsafe_missing or shape_mismatches:
        raise RuntimeError(
            f"Checkpoint {selected.source_path} is not compatible with translated model.\n"
            f"Missing: {missing[:20]}\n"
            f"Unexpected: {unexpected[:20]}\n"
            f"Shape mismatches: {shape_mismatches[:20]}"
        )

    model.load_state_dict(state_dict, strict=False)
    epoch = _loaded_epoch(loaded)
    if epoch is None:
        epoch = selected.epoch
    payload = {
        "epoch": -1 if epoch is None else int(epoch),
        "model_state_dict": model.state_dict(),
    }
    if isinstance(loaded, dict):
        for key in ("best_loss", "train_loss", "val_loss"):
            if key in loaded:
                payload[key] = loaded[key]
    return ConvertedCheckpoint(selected.selector, selected.source_path, epoch, payload), model


def _instantiate_model(model_spec: LegacyModelSpec, paths: Paths):
    import torch

    from lisai.models.load_nm import load_noise_model
    from lisai.models.loader import init_model

    device = torch.device("cpu")
    noise_model = None
    if model_spec.architecture == "lvae":
        noise_model, _ = load_noise_model(model_spec.noise_model_name, device, paths)

    return init_model(
        architecture=model_spec.architecture,
        model_prm=model_spec.parameters,
        device=device,
        model_norm_prm=model_spec.model_norm_prm,
        noise_model=noise_model,
        img_shape=None if model_spec.runtime_img_shape is None else model_spec.runtime_img_shape[0],
    )


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save(payload: dict[str, Any], path: Path):
    import torch

    torch.save(payload, path)


def _extract_model_state_dict(loaded, checkpoint_path: Path) -> dict[str, Any]:
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    elif isinstance(loaded, dict):
        state_dict = loaded
    else:
        raise ValueError(f"Unsupported checkpoint type at {checkpoint_path}: {type(loaded)!r}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Model state dict at {checkpoint_path} must be a dict.")
    return state_dict


def _state_dict_diff(model, legacy_state_dict: dict[str, Any]):
    current_state = model.state_dict()
    missing = sorted(set(current_state) - set(legacy_state_dict))
    unexpected = sorted(set(legacy_state_dict) - set(current_state))
    shape_mismatches = []
    for key in sorted(set(current_state) & set(legacy_state_dict)):
        current_value = current_state[key]
        legacy_value = legacy_state_dict[key]
        if hasattr(current_value, "shape") and hasattr(legacy_value, "shape"):
            if tuple(current_value.shape) != tuple(legacy_value.shape):
                shape_mismatches.append(
                    (key, tuple(legacy_value.shape), tuple(current_value.shape))
                )
    return missing, unexpected, shape_mismatches


def _is_allowed_inactive_missing_key(key: str, model_spec: LegacyModelSpec) -> bool:
    if model_spec.architecture == "unet":
        params = model_spec.parameters
        if key.startswith("upsamp_before."):
            return not (params.upsampling_factor > 1 and params.upsampling_order == "before")
        if key.startswith("upsamp_after"):
            return not (params.upsampling_factor > 1 and params.upsampling_order == "after")
    if model_spec.architecture == "unet_rcan":
        return key.startswith("unet.upsamp_before.")
    return False


def build_import_metadata(
    *,
    job: LegacyImportJob,
    target_run_dir: Path,
    resolved_cfg: ResolvedExperiment,
    selected_checkpoints: tuple[ConvertedCheckpoint, ...],
    best_val_loss: float | None,
    trainable_params: int | None,
) -> RunMetadata:
    now = utc_now()
    run_name, run_index = parse_run_dir_name(target_run_dir.name)
    last_epoch = _max_int_or_none(checkpoint.epoch for checkpoint in selected_checkpoints)
    max_epoch = _max_int_or_none(
        [
            last_epoch,
            getattr(resolved_cfg.training, "n_epochs", None),
        ]
    )
    training_signature = build_training_signature_from_resolved_config(
        resolved_cfg,
        trainable_params=trainable_params,
    )

    return RunMetadata(
        run_id=generate_run_id(),
        run_name=run_name,
        run_index=run_index,
        dataset=job.target_dataset,
        model_subfolder=job.target_model_subfolder,
        status="completed",
        closed_cleanly=True,
        created_at=now,
        updated_at=now,
        ended_at=now,
        last_heartbeat_at=now,
        last_epoch=last_epoch,
        max_epoch=max_epoch,
        best_val_loss=best_val_loss,
        path=stored_run_path(target_run_dir),
        group_path=group_path_from_model_subfolder(job.target_model_subfolder),
        training_signature=TrainingSignature.model_validate(training_signature),
        provenance=RunProvenance(
            source_path=str(job.legacy_model_path.resolve()),
            source_config="config_train.json",
            imported_at=now,
            notes="Imported by src/scripts/import_legacy_models.py",
        ),
    )

def execute_import(plan: ImportPlan):
    paths = Paths(settings)
    if plan.target_run_dir.exists():
        raise FileExistsError(f"Target run folder already exists: {plan.target_run_dir}")

    ensure_folder(plan.target_run_dir.parent, mode="exist_ok")
    ensure_folder(plan.target_run_dir, mode="strict")
    checkpoints_dir = paths.checkpoints_dir(run_dir=plan.target_run_dir)
    ensure_folder(checkpoints_dir, mode="strict")

    save_yaml(
        prune_config_for_saving(plan.resolved_cfg),
        paths.cfg_train_path(run_dir=plan.target_run_dir),
    )
    for checkpoint in plan.checkpoints:
        target_name = model_filename(load_method="state_dict", best_or_last=checkpoint.selector)
        _torch_save(checkpoint.payload, checkpoints_dir / target_name)

    root_artifacts = _copy_run_root_legacy_artifacts(
        plan.job.legacy_model_path,
        plan.target_run_dir,
        paths,
    )
    if "loss.txt" in root_artifacts:
        _save_imported_loss_plot(plan, paths)

    if plan.job.copy_small_legacy_artifacts:
        _copy_legacy_origin_artifacts(
            plan.job.legacy_model_path,
            plan.target_run_dir,
            copied_run_root_artifacts=root_artifacts,
        )

    write_run_metadata_atomic(plan.target_run_dir, plan.metadata)


def _copy_run_root_legacy_artifacts(
    legacy_model_path: Path,
    target_run_dir: Path,
    paths: Paths,
) -> set[str]:
    copied: set[str] = set()
    artifact_targets = {
        "loss.txt": paths.loss_file_path(run_dir=target_run_dir),
        "train_log.log": paths.log_file_path(run_dir=target_run_dir),
    }
    for name, dst in artifact_targets.items():
        src = legacy_model_path / name
        if src.exists():
            shutil.copyfile(src, dst)
            copied.add(name)
    return copied


def _save_imported_loss_plot(plan: ImportPlan, paths: Paths):
    architecture = plan.resolved_cfg.model.architecture if plan.resolved_cfg.model else None
    try:
        saved_path = save_loss_plot_for_run(
            run_dir=plan.target_run_dir,
            dataset=plan.job.target_dataset,
            model_subfolder=plan.job.target_model_subfolder,
            architecture=architecture,
            paths=paths,
            stderr=sys.stderr,
        )
    except Exception as exc:
        print(
            f"warning: imported loss plot was not saved: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return

    if saved_path is None:
        print("warning: imported loss plot was not saved.", file=sys.stderr)


def _copy_legacy_origin_artifacts(
    legacy_model_path: Path,
    target_run_dir: Path,
    *,
    copied_run_root_artifacts: set[str],
):
    legacy_origin = target_run_dir / "legacy_origin"
    ensure_folder(legacy_origin, mode="strict")
    copied_legacy_origin_artifacts: list[str] = []
    for name in ("config_train.json",):
        src = legacy_model_path / name
        if src.exists():
            shutil.copyfile(src, legacy_origin / name)
            copied_legacy_origin_artifacts.append(name)

    summary = {
        "source_path": str(legacy_model_path.resolve()),
        "copied_legacy_origin_artifacts": sorted(copied_legacy_origin_artifacts),
        "copied_run_root_artifacts": sorted(copied_run_root_artifacts),
    }
    with (legacy_origin / "import_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def print_import_plan(plan: ImportPlan):
    print("Legacy import preflight succeeded.")
    print(f"  Source: {plan.job.legacy_model_path}")
    print(f"  Target: {plan.target_run_dir}")
    architecture = plan.resolved_cfg.model.architecture if plan.resolved_cfg.model else "unknown"
    print(f"  Architecture: {architecture}")
    print(f"  Dataset: {plan.job.target_dataset}")
    print(f"  Model subfolder: {plan.job.target_model_subfolder}")
    print(f"  Run name: {plan.job.target_run_name}")
    print(f"  Trainable params: {plan.trainable_params}")
    print(f"  Best val loss: {plan.best_val_loss}")
    print("  Checkpoints:")
    for checkpoint in plan.checkpoints:
        target_name = model_filename(load_method="state_dict", best_or_last=checkpoint.selector)
        print(
            f"    {checkpoint.selector}: {checkpoint.source_path.name}"
            f" -> checkpoints/{target_name}"
            f" (epoch={checkpoint.epoch})"
        )
    if plan.model_spec.noise_model_name:
        nm_path = Paths(settings).noise_model_path(noiseModel_name=plan.model_spec.noise_model_name)
        print(f"  Noise model: {plan.model_spec.noise_model_name} ({nm_path})")


def _best_epoch_from_loss(loss_path: Path) -> int | None:
    rows = _loss_rows(loss_path)
    if not rows:
        return None
    return min(rows, key=lambda row: row[1])[0]


def _best_val_loss(loss_path: Path) -> float | None:
    rows = _loss_rows(loss_path)
    if not rows:
        return None
    return min(rows, key=lambda row: row[1])[1]


def _loss_rows(loss_path: Path) -> list[tuple[int, float]]:
    if not loss_path.exists():
        return []
    rows: list[tuple[int, float]] = []
    for line in loss_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), float(parts[2])))
        except ValueError:
            continue
    return rows


def _loaded_epoch(loaded) -> int | None:
    if isinstance(loaded, dict) and loaded.get("epoch") is not None:
        try:
            return int(loaded["epoch"])
        except (TypeError, ValueError):
            return None
    return None


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"model_epoch_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def _max_int_or_none(values) -> int | None:
    parsed: list[int] = []
    for value in values:
        if value is None:
            continue
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(parsed) if parsed else None


if __name__ == "__main__":
    raise SystemExit(main())
