"""High-level entrypoint for dataset-based evaluation of a saved model.

This module orchestrates the full evaluation flow for an already-resolved typed
``EvaluateDefaults`` config: resolve the saved run, initialize the inference
runtime, rebuild the evaluation sample source, run predictions, compute metrics,
and persist outputs.
"""

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

from lisai.config import save_yaml
from lisai.config.models.inference import EvaluateDefaults
from lisai.config.progress import resolve_progress_bar
from lisai.evaluation.data import (
    EvaluationDatasetSpec,
    build_eval_source,
    resolve_evaluation_dataset,
)
from lisai.evaluation.inference.progress import InferenceProgress
from lisai.evaluation.inference.stack import infer_batch
from lisai.evaluation.io import EvalItemOutputWriter, save_metrics_json
from lisai.evaluation.metrics import compute as metrics
from lisai.evaluation.runtime import TilingSizePolicy, initialize_runtime
from lisai.evaluation.saved_run import SavedTrainingRun, load_saved_run, resolve_run_dir
from lisai.infra.fs import prepare_output_folder


def _build_evaluation_folder_name(
    *,
    best_or_last: str,
    requested_epoch: int | None,
    resolved_epoch: int | None,
) -> str:
    """Build the checkpoint-specific subfolder for one evaluation call."""
    if requested_epoch is not None:
        return f"epoch_{requested_epoch}"
    if resolved_epoch is not None:
        return f"{best_or_last}_epoch_{resolved_epoch}"
    return best_or_last


def _safe_evaluation_dataset_folder(name: str) -> str:
    """Keep dataset names readable while preventing accidental nested paths."""
    return str(name).replace("\\", "__").replace("/", "__")


def _evaluation_dataset_folder(
    *,
    evaluation_dataset: EvaluationDatasetSpec | None,
    split: str,
) -> str:
    if evaluation_dataset is not None:
        return _safe_evaluation_dataset_folder(evaluation_dataset.name)
    return f"training_{split}"


def _evaluation_metadata(
    *,
    saved_run: SavedTrainingRun,
    runtime,
    sample_source,
    cfg: EvaluateDefaults,
    evaluation_dataset: EvaluationDatasetSpec | None,
) -> dict[str, Any]:
    """Build lightweight provenance for one persisted evaluation result."""
    if evaluation_dataset is None:
        dataset_name = saved_run.dataset_name
        dataset_usage = "training"
        data_type = None
        split = cfg.data.split
    else:
        dataset_name = evaluation_dataset.name
        dataset_usage = "evaluation"
        data_type = evaluation_dataset.data_type
        split = None

    return {
        "version": 1,
        "model": {
            "source": "training_run",
            "run_dir": str(saved_run.run_dir),
            "experiment_name": saved_run.experiment_name,
            "training_dataset": saved_run.dataset_name,
            "checkpoint": {
                "selector": cfg.checkpoint.best_or_last,
                "requested_epoch": cfg.checkpoint.epoch_number,
                "resolved_epoch": runtime.resolved_epoch,
                "load_method": runtime.load_method,
                "path": str(runtime.checkpoint_path),
            },
        },
        "dataset": {
            "name": dataset_name,
            "usage": dataset_usage,
            "data_type": data_type,
            "split": split,
            "data_dir": str(sample_source.config.data_dir),
            "input": sample_source.config.input,
            "eval_gt": sample_source.config.target,
            "data_format": sample_source.config.resolved_data_format,
        },
        "evaluation": {
            "metrics": cfg.metrics,
            "tiling_size": {
                "requested": cfg.inference.tiling_size,
                "effective": runtime.tiling_size,
            },
        },
    }


def _expand_checkpoint_selection(cfg: EvaluateDefaults) -> list[EvaluateDefaults]:
    """Expand ``best_or_last='both'`` into independent typed checkpoint configs."""
    if cfg.checkpoint.epoch_number is not None or cfg.checkpoint.best_or_last != "both":
        return [cfg]

    expanded: list[EvaluateDefaults] = []
    base_save_folder = cfg.saving.save_folder
    for selector in ("best", "last"):
        values = cfg.model_dump()
        values["checkpoint"]["best_or_last"] = selector
        if base_save_folder is not None:
            values["saving"]["save_folder"] = str(Path(base_save_folder) / selector)
        expanded.append(EvaluateDefaults.model_validate(values))
    return expanded


def _format_eval_gt_for_display(eval_gt: str | None) -> str:
    if eval_gt is None:
        return "<none>"
    if eval_gt == "":
        return "<root>"
    return str(eval_gt)


def _format_tiling_size_for_display(requested: TilingSizePolicy, effective: int | None) -> str:
    if effective is None:
        return "off"
    if requested is None or requested == "auto":
        return f"{effective} (auto)"
    return str(effective)


def _resolved_data_overrides(cfg: EvaluateDefaults) -> dict[str, Any] | None:
    """Build per-run data overrides without mutating the resolved config."""
    data_overrides = deepcopy(cfg.data.overrides)
    if cfg.data.timelapse_max is None:
        return data_overrides

    data_overrides = data_overrides or {}
    timelapse_prm = dict(data_overrides.get("timelapse_prm") or {})
    timelapse_prm["timelapse_max_frames"] = cfg.data.timelapse_max
    data_overrides["timelapse_prm"] = timelapse_prm
    return data_overrides


def _run_single_evaluation(
    *,
    output_root: Path,
    saved_run: SavedTrainingRun,
    cfg: EvaluateDefaults,
    progress_bar: bool,
    evaluation_dataset: EvaluationDatasetSpec | None = None,
) -> None:
    """Run one resolved checkpoint evaluation and save outputs/metrics."""
    runtime = initialize_runtime(
        saved_run=saved_run,
        best_or_last=cfg.checkpoint.best_or_last,
        epoch_number=cfg.checkpoint.epoch_number,
        tiling_size=cfg.inference.tiling_size,
    )

    if cfg.saving.save_folder is None:
        dataset_folder = _evaluation_dataset_folder(
            evaluation_dataset=evaluation_dataset,
            split=cfg.data.split,
        )
        checkpoint_folder = _build_evaluation_folder_name(
            best_or_last=cfg.checkpoint.best_or_last,
            requested_epoch=cfg.checkpoint.epoch_number,
            resolved_epoch=runtime.resolved_epoch,
        )
        resolution = prepare_output_folder(
            path=Path(output_root) / dataset_folder / checkpoint_folder,
            if_exists_policy="overwrite" if cfg.saving.overwrite else "numbered",
        )
    else:
        resolution = prepare_output_folder(
            Path(cfg.saving.save_folder),
            if_exists_policy="reuse",
        )

    save_folder = resolution.path
    print(resolution.message())

    if saved_run.is_lvae:
        assert cfg.inference.lvae_num_samples is not None, (
            "for LVAE prediction, number of samples needs to be specified"
        )

    progress = InferenceProgress(enabled=progress_bar)

    upsamp = saved_run.upsampling_factor
    print(f"Found upsampling factor to be: {upsamp}\n")
    tiling_size = runtime.tiling_size
    print(
        "Tiling size: "
        f"{_format_tiling_size_for_display(cfg.inference.tiling_size, tiling_size)}\n"
    )

    data_overrides = _resolved_data_overrides(cfg)
    sample_source = build_eval_source(
        saved_run,
        split=cfg.data.split,
        crop_size=cfg.inference.crop_size,
        eval_gt=cfg.data.eval_gt,
        data_overrides=data_overrides,
        evaluation_dataset=evaluation_dataset,
    )
    print(f"Evaluation GT: {_format_eval_gt_for_display(sample_source.config.target)}")
    save_yaml(
        _evaluation_metadata(
            saved_run=saved_run,
            runtime=runtime,
            sample_source=sample_source,
            cfg=cfg,
            evaluation_dataset=evaluation_dataset,
        ),
        save_folder / "evaluation.yaml",
    )
    results = None

    limit_n_imgs = cfg.data.limit_n_imgs
    if limit_n_imgs is not None:
        n_total = min(len(sample_source), limit_n_imgs)
        if n_total < len(sample_source):
            print(
                f"Found {len(sample_source)} images, keeping only "
                f"{limit_n_imgs} because of args limit_n_imgs."
            )
        else:
            print(f"Found {len(sample_source)} images.")
    else:
        n_total = len(sample_source)

    n_processed = 0
    stop_eval = False
    for item_id, item in enumerate(sample_source.iter_items()):
        if item.data_format == "timelapse":
            n_timepoints = len(item)
            print(
                f"Item {item_id} / {len(sample_source.items)} -"
                f" {n_timepoints} frames: {item.name}"
            )
        else:
            print(f"Item {item_id} / {n_total}: {item.name}")

        writer = EvalItemOutputWriter(item=item, save_folder=save_folder)
        for sample_index, sample in item.iter_samples(sample_source.config):
            if item.data_format == "timelapse":
                print(
                    f"Item {item_id} - frame {sample_index}/{n_timepoints} "
                    f"(total evaluation images: {n_processed}/{len(sample_source)})"
                )
            else:
                print(f"Image {n_processed} / {len(sample_source)}")

            x = sample.x.unsqueeze(0)
            y = sample.y.unsqueeze(0) if sample.y is not None else None
            x = x.to(runtime.device)
            print(f"Input shape: {x.shape}")
            resolved_ch_out = cfg.inference.ch_out
            if resolved_ch_out is None and x.ndim >= 4 and x.shape[1] > 1:
                # Context/multi-channel inputs are used to predict one target frame.
                resolved_ch_out = 1

            outputs = infer_batch(
                runtime.model,
                x,
                is_lvae=saved_run.is_lvae,
                tiling_size=tiling_size,
                num_samples=cfg.inference.lvae_num_samples,
                upsamp=upsamp,
                ch_out=resolved_ch_out,
                progress=progress,
            )
            img_name = item.sample_name(sample_index)
            tosave = {
                "inp": x.cpu().detach().numpy(),
                "gt": y.cpu().detach().numpy() if y is not None else None,
                "pred": outputs.get("prediction"),
                "samples": outputs.get("samples"),
            }
            writer.add(sample_index=sample_index, tosave=tosave)

            if cfg.metrics is not None and y is None:
                warnings.warn("no ground-truth provided, cannot calculate metrics")
            elif cfg.metrics is not None:
                if x.shape[-2:] == y.shape[-2:]:
                    inp = x.cpu().detach().numpy()
                else:
                    inp = None

                gt = y.cpu().detach().numpy()
                results = metrics.calculate_metrics(
                    img_name=img_name,
                    metrics=cfg.metrics,
                    results=results,
                    pred=outputs.get("prediction"),
                    gt=gt,
                    inp=inp,
                )

            n_processed += 1
            if limit_n_imgs is not None and n_processed >= limit_n_imgs:
                print("Stopping eval because reached limit_n_imgs")
                stop_eval = True
                break

        writer.flush()
        if stop_eval:
            break

    if cfg.metrics is not None and results is not None:
        save_metrics_json(save_folder, results)


def run_evaluate(
    cfg: EvaluateDefaults,
    *,
    dataset_name: str,
    model_name: str,
    model_subfolder: str = "",
    evaluation_dataset_name: str | None = None,
    progress_bar: bool | None = None,
) -> None:
    """Evaluate a saved run using a fully resolved typed evaluate config."""
    resolved_progress_bar = resolve_progress_bar(True, progress_bar)

    run_dir = resolve_run_dir(
        dataset_name=dataset_name,
        subfolder=model_subfolder,
        exp_name=model_name,
    )
    saved_run = load_saved_run(run_dir)
    evaluation_dataset = None
    if evaluation_dataset_name is not None:
        evaluation_dataset = resolve_evaluation_dataset(
            saved_run,
            evaluation_dataset_name,
            data_overrides=cfg.data.overrides,
        )
        print(f"Evaluation dataset: {evaluation_dataset.name} (all data)")

    run_configs = _expand_checkpoint_selection(cfg)
    if len(run_configs) > 1:
        print("best_or_last='both': running evaluation for checkpoints ['best', 'last'].")

    output_root = run_dir / "evaluations"
    for run_cfg in run_configs:
        _run_single_evaluation(
            output_root=output_root,
            saved_run=saved_run,
            cfg=run_cfg,
            progress_bar=resolved_progress_bar,
            evaluation_dataset=evaluation_dataset,
        )
