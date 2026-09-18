from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lisai.config.models.inference import EvaluateDefaults, EvaluateOverrides
from lisai.config.settings import settings
from lisai.evaluation.defaults import resolve_evaluate_config
from lisai.evaluation.run_evaluate import run_evaluate
from lisai.infra.paths import Paths

dataset = "vim_fixed_multi_snr"
model_subfolder = "HDN"

inference_config: str | Path | None = "default"

evaluation_dataset_name: str | None = None  # equivalent of `lisai evaluate --on DATASET`.
split: str | None = "train"                  # if evaluation_dataset_name is None.
eval_gt: str | None = None

# Checkpoint options. Leave as None to keep the value from the inference config.
best_or_last: str | None = "best" 
epoch_number: int | None = None

# Runtime/evaluation options. Leave as None to keep the value from the inference config.
metrics: list[str] | None = None  
tiling_size: int | str | None = None  
crop_size: int | tuple[int, int] | None = None
lvae_num_samples: int | None = None
ch_out: int | None = None
limit_n_imgs: int | None = None
timelapse_max: int | None = None
overwrite: bool | None = True
progress_bar: bool | None = None

# Run-folder filtering.
skip_if_contain: list[str] = []
skip_folders: list[str] = ["hdn_sup_betaKL001_00"]
case_sensitive_skip = False


# commmon saving option: to save under a common save folder 
common_save_location = False
common_save_folder: str | Path = r""
include_checkpoint_in_common_folder_name = True

# Safety and error handling.
dry_run = False
stop_on_error = False


@dataclass(frozen=True)
class PlannedEvaluation:
    run_dir: Path
    save_folder: Path | None


def main() -> int:
    paths = Paths(settings)
    base_cfg = _resolve_base_config()
    run_dirs = _discover_run_dirs(paths)

    if not run_dirs:
        print(
            "No run folders found for "
            f"dataset={dataset!r}, model_subfolder={_normalized_model_subfolder()!r}."
        )
        return 1

    planned = [
        PlannedEvaluation(
            run_dir=run_dir,
            save_folder=_common_save_folder(run_dir.name, base_cfg) if common_save_location else None,
        )
        for run_dir in run_dirs
    ]

    _print_plan(planned, base_cfg)
    if dry_run:
        print("\nDRY RUN: set dry_run = False at the top of this script to run evaluation.")
        return 0

    failures: list[tuple[Path, Exception]] = []
    for index, item in enumerate(planned, start=1):
        print(f"\n[{index}/{len(planned)}] Evaluating {item.run_dir.name}")
        cfg = _config_for_run(base_cfg, save_folder=item.save_folder)
        try:
            run_evaluate(
                cfg=cfg,
                dataset_name=dataset,
                model_subfolder=_normalized_model_subfolder(),
                model_name=item.run_dir.name,
                evaluation_dataset_name=evaluation_dataset_name,
                progress_bar=progress_bar,
            )
        except Exception as exc:
            print(f"FAILED: {item.run_dir.name}: {type(exc).__name__}: {exc}")
            failures.append((item.run_dir, exc))
            if stop_on_error:
                raise

    print("\nFinished batch evaluation.")
    print(f"Successful runs: {len(planned) - len(failures)}")
    print(f"Failed runs: {len(failures)}")
    if failures:
        for run_dir, exc in failures:
            print(f"  - {run_dir.name}: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _resolve_base_config() -> EvaluateDefaults:
    cfg = resolve_evaluate_config(
        config=_normalized_inference_config(),
        overrides=_script_overrides(),
    )
    if common_save_location:
        if common_save_folder is None or str(common_save_folder).strip() == "":
            raise ValueError("common_save_folder must be set when common_save_location is True.")
        return cfg
    return _config_for_run(cfg, save_folder=None)


def _script_overrides() -> EvaluateOverrides:
    payload: dict[str, Any] = {}

    checkpoint: dict[str, Any] = {}
    if epoch_number is not None:
        checkpoint["epoch_number"] = epoch_number
    if best_or_last is not None:
        checkpoint["best_or_last"] = best_or_last
    if checkpoint:
        payload["checkpoint"] = checkpoint

    data: dict[str, Any] = {}
    if evaluation_dataset_name is None and split is not None:
        data["split"] = split
    if eval_gt is not None:
        data["eval_gt"] = eval_gt
    if limit_n_imgs is not None:
        data["limit_n_imgs"] = limit_n_imgs
    if timelapse_max is not None:
        data["timelapse_max"] = timelapse_max
    if data:
        payload["data"] = data

    inference: dict[str, Any] = {}
    if tiling_size is not None:
        inference["tiling_size"] = tiling_size
    if crop_size is not None:
        inference["crop_size"] = crop_size
    if lvae_num_samples is not None:
        inference["lvae_num_samples"] = lvae_num_samples
    if ch_out is not None:
        inference["ch_out"] = ch_out
    if inference:
        payload["inference"] = inference

    if metrics is not None:
        payload["metrics"] = metrics

    saving: dict[str, Any] = {}
    if overwrite is not None:
        saving["overwrite"] = overwrite
    if saving:
        payload["saving"] = saving

    return EvaluateOverrides.model_validate(payload)


def _config_for_run(cfg: EvaluateDefaults, *, save_folder: Path | None) -> EvaluateDefaults:
    values = cfg.model_dump()
    values["saving"]["save_folder"] = None if save_folder is None else str(save_folder)
    return EvaluateDefaults.model_validate(values)


def _discover_run_dirs(paths: Paths) -> list[Path]:
    root = paths.dataset_runs_dir(dataset_name=dataset) / Path(_normalized_model_subfolder())
    if not root.exists():
        raise FileNotFoundError(f"Model subfolder does not exist: {root}")

    run_dirs: list[Path] = []
    skipped: list[str] = []
    for candidate in sorted((path for path in root.iterdir() if path.is_dir()), key=_natural_path_key):
        if _skip_run_folder(candidate.name):
            skipped.append(candidate.name)
            continue
        if not paths.cfg_train_path(run_dir=candidate).is_file():
            skipped.append(f"{candidate.name} (missing config_train.yaml)")
            continue
        run_dirs.append(candidate)

    print(f"Run root: {root}")
    if skipped:
        print("Skipped folders:")
        for name in skipped:
            print(f"  - {name}")
    return run_dirs


def _skip_run_folder(name: str) -> bool:
    haystack = name if case_sensitive_skip else name.casefold()
    exact_skip = _normalized_skip_values(skip_folders)
    contains_skip = _normalized_skip_values(skip_if_contain)
    return haystack in exact_skip or any(part in haystack for part in contains_skip)


def _normalized_skip_values(values: list[str]) -> list[str]:
    cleaned = [item.strip() for item in values if item.strip()]
    if case_sensitive_skip:
        return cleaned
    return [item.casefold() for item in cleaned]


def _common_save_folder(run_name: str, cfg: EvaluateDefaults) -> Path:
    parts = [run_name, _evaluation_source_label(cfg)]
    if include_checkpoint_in_common_folder_name:
        parts.append(_checkpoint_label(cfg))
    folder_name = "_".join(_safe_path_part(part) for part in parts if part)
    return Path(common_save_folder).expanduser() / folder_name


def _evaluation_source_label(cfg: EvaluateDefaults) -> str:
    if evaluation_dataset_name is not None:
        return _safe_path_part(evaluation_dataset_name)
    return f"training_{cfg.data.split}"


def _checkpoint_label(cfg: EvaluateDefaults) -> str:
    if cfg.checkpoint.epoch_number is not None:
        return f"epoch_{cfg.checkpoint.epoch_number}"
    return str(cfg.checkpoint.best_or_last)


def _safe_path_part(value: str) -> str:
    text = str(value).replace("\\", "/").strip("/")
    text = re.sub(r"[^A-Za-z0-9._=-]+", "_", text)
    return text.strip("_") or "unnamed"


def _natural_path_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def _normalized_model_subfolder() -> str:
    return str(model_subfolder).replace("\\", "/").strip("/")


def _normalized_inference_config() -> str | Path | None:
    if inference_config is None:
        return None
    if isinstance(inference_config, str):
        text = inference_config.strip()
        if text.casefold() in {"", "default", "defaults", "local/defaults"}:
            return None
        return text
    return inference_config


def _print_plan(planned: list[PlannedEvaluation], cfg: EvaluateDefaults) -> None:
    print("\nEvaluation plan")
    print(f"  dataset: {dataset}")
    print(f"  model_subfolder: {_normalized_model_subfolder()}")
    print(f"  inference_config: {inference_config!r}")
    print(f"  source: {_evaluation_source_label(cfg)}")
    print(
        "  checkpoint: "
        f"best_or_last={cfg.checkpoint.best_or_last}, epoch_number={cfg.checkpoint.epoch_number}"
    )
    print(f"  metrics: {cfg.metrics}")
    print(f"  common_save_location: {common_save_location}")
    if common_save_location:
        print(f"  common_save_folder: {Path(common_save_folder).expanduser()}")
    print(f"  dry_run: {dry_run}")
    print(f"\nRuns to evaluate ({len(planned)}):")
    for item in planned:
        if item.save_folder is None:
            destination = "normal per-run evaluation folder"
        else:
            destination = str(item.save_folder)
        print(f"  - {item.run_dir.name} -> {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
