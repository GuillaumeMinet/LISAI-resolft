"""High-level entrypoint for dataset-based evaluation of a saved model.

This module orchestrates the full evaluation flow: resolve the saved run,
initialize the inference runtime, rebuild the evaluation sample source,
run predictions, compute metrics, and persist outputs.
"""

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

from lisai.config import save_yaml
from lisai.evaluation.data import (
    EvaluationDatasetSpec,
    build_eval_source,
    resolve_evaluation_dataset,
)
from lisai.evaluation.defaults import UNSET, UnsetType, resolve_evaluate_options
from lisai.evaluation.inference.stack import infer_batch
from lisai.evaluation.io import (
    EvalItemOutputWriter,
    create_save_folder,
    ensure_save_folder,
    save_metrics_json,
)
from lisai.evaluation.metrics import compute as metrics
from lisai.evaluation.runtime import TilingSizePolicy, initialize_runtime
from lisai.evaluation.saved_run import SavedTrainingRun, load_saved_run, resolve_run_dir


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
    options: dict[str, Any],
    evaluation_dataset: EvaluationDatasetSpec | None,
) -> dict[str, Any]:
    """Build lightweight provenance for one persisted evaluation result."""
    if evaluation_dataset is None:
        dataset_name = saved_run.dataset_name
        dataset_usage = "training"
        data_type = None
        split = options["split"]
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
                "selector": options["best_or_last"],
                "requested_epoch": options["epoch_number"],
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
            "metrics": options["metrics_list"],
            "tiling_size": {
                "requested": options["tiling_size"],
                "effective": runtime.tiling_size,
            },
        },
    }


def _expand_checkpoint_selection(options: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand `best_or_last='both'` into independent checkpoint runs."""
    if options["epoch_number"] is not None or options["best_or_last"] != "both":
        return [options]

    expanded: list[dict[str, Any]] = []
    base_save_folder = options["save_folder"]
    for selector in ("best", "last"):
        selector_options = dict(options)
        selector_options["best_or_last"] = selector
        if isinstance(options.get("results"), dict):
            selector_options["results"] = deepcopy(options["results"])
        if base_save_folder is not None:
            selector_options["save_folder"] = Path(base_save_folder) / selector
        expanded.append(selector_options)
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


def _run_single_evaluation(
    *,
    output_root: Path,
    saved_run: SavedTrainingRun,
    options: dict[str, Any],
    evaluation_dataset: EvaluationDatasetSpec | None = None,
) -> None:
    """Run one resolved checkpoint evaluation and save outputs/metrics."""
    runtime = initialize_runtime(
        saved_run=saved_run,
        best_or_last=options["best_or_last"],
        epoch_number=options["epoch_number"],
        tiling_size=options["tiling_size"],
    )

    if options["save_folder"] is None:
        dataset_folder = _evaluation_dataset_folder(
            evaluation_dataset=evaluation_dataset,
            split=options["split"],
        )
        checkpoint_folder = _build_evaluation_folder_name(
            best_or_last=options["best_or_last"],
            requested_epoch=options["epoch_number"],
            resolved_epoch=runtime.resolved_epoch,
        )
        save_folder = create_save_folder(
            path=Path(output_root) / dataset_folder / checkpoint_folder,
            overwrite=options["overwrite"],
        )
    else:
        save_folder = ensure_save_folder(Path(options["save_folder"]))

    if save_folder is None:
        raise FileNotFoundError("Could not create evaluation output folder.")

    if saved_run.is_lvae:
        assert options["lvae_num_samples"] is not None, (
            "for LVAE prediction, number of samples needs to be specified"
        )

    upsamp = saved_run.upsampling_factor
    print(f"Found upsampling factor to be: {upsamp}\n")
    tiling_size = runtime.tiling_size
    print(f"Tiling size: {_format_tiling_size_for_display(options['tiling_size'], tiling_size)}\n")

    # inject timelapse_max option into data_prm_update
    timelapse_max = options.get("timelapse_max")
    if timelapse_max is not None:
        data_prm_update = deepcopy(options["data_prm_update"] or {})
        timelapse_prm = dict(data_prm_update.get("timelapse_prm") or {})
        timelapse_prm["timelapse_max_frames"] = options["timelapse_max"]
        data_prm_update["timelapse_prm"] = timelapse_prm
        options["data_prm_update"] = data_prm_update

    sample_source = build_eval_source(
        saved_run,
        split=options["split"],
        crop_size=options["crop_size"],
        eval_gt=options["eval_gt"],
        data_prm_update=options["data_prm_update"],
        evaluation_dataset=evaluation_dataset,
    )
    print(f"Evaluation GT: {_format_eval_gt_for_display(sample_source.config.target)}")
    save_yaml(
        _evaluation_metadata(
            saved_run=saved_run,
            runtime=runtime,
            sample_source=sample_source,
            options=options,
            evaluation_dataset=evaluation_dataset,
        ),
        save_folder / "evaluation.yaml",
    )
    results = options["results"]

    limit_n_imgs = options["limit_n_imgs"]
    if limit_n_imgs is not None:
        n_total = min(len(sample_source),limit_n_imgs)
        if n_total < len(sample_source):
            print(f"Found {len(sample_source)} images, keeping only "
                  f"{limit_n_imgs} because of args limit_n_imgs.")
        else:
            print(f"Found {len(sample_source)} images.")
    else:
        n_total = len(sample_source)

    n_processed = 0
    stop_eval = False
    for item_id, item in enumerate(sample_source.iter_items()):
        if item.data_format == "timelapse":
            n_timepoints = len(item)
            print(f"Item {item_id} / {len(sample_source.items)} -"
                  f" {n_timepoints} frames: {item.name}")
        else:
            print(f"Item {item_id} / {n_total}: {item.name}")

        writer = EvalItemOutputWriter(item=item, save_folder=save_folder)
        for sample_index, sample in item.iter_samples(sample_source.config):
            if item.data_format == "timelapse":
                print(f"Item {item_id} - frame {sample_index}/{n_timepoints} "
                      f"(total evaluation images: {n_processed}/{len(sample_source)})")
            else:
                print(f"Image {n_processed} / {len(sample_source)}")

            x = sample.x.unsqueeze(0)
            y = sample.y.unsqueeze(0) if sample.y is not None else None
            x = x.to(runtime.device)
            print(f"Input shape: {x.shape}")
            resolved_ch_out = options["ch_out"]
            if resolved_ch_out is None and x.ndim >= 4 and x.shape[1] > 1:
                # Context/multi-channel inputs are used to predict one target frame.
                resolved_ch_out = 1

            outputs = infer_batch(
                runtime.model,
                x,
                is_lvae=saved_run.is_lvae,
                tiling_size=tiling_size,
                num_samples=options["lvae_num_samples"],
                upsamp=upsamp,
                ch_out=resolved_ch_out,
            )
            img_name = item.sample_name(sample_index)
            tosave = {
                "inp": x.cpu().detach().numpy(),
                "gt": y.cpu().detach().numpy() if y is not None else None,
                "pred": outputs.get("prediction"),
                "samples": outputs.get("samples"),
            }
            writer.add(sample_index=sample_index, tosave=tosave)

            if options["metrics_list"] is not None and y is None:
                warnings.warn("no ground-truth provided, cannot calculate metrics")
            elif options["metrics_list"] is not None:
                if x.shape[-2:] == y.shape[-2:]:
                    inp = x.cpu().detach().numpy()
                else:
                    inp = None

                gt = y.cpu().detach().numpy()
                results = metrics.calculate_metrics(
                    img_name=img_name,
                    metrics=options["metrics_list"],
                    results=results,
                    pred=outputs.get("prediction"),
                    gt=gt,
                    inp=inp,
                )

            n_processed += 1
            if options["limit_n_imgs"] is not None and n_processed >= options["limit_n_imgs"]:
                print("Stopping eval because reached limit_n_imgs")
                stop_eval = True
                break

        writer.flush()
        if stop_eval:
            break

    if options["metrics_list"] is not None and results is not None:
        save_metrics_json(save_folder, results)


def run_evaluate(dataset_name:str,
             model_name:str,
             model_subfolder:str="",
             best_or_last: str | UnsetType = UNSET,
             epoch_number: int | None | UnsetType = UNSET,
             tiling_size: TilingSizePolicy | UnsetType = UNSET,
             crop_size: int | tuple[int, int] | None | UnsetType = UNSET,
             metrics_list: list[str] | None | UnsetType = UNSET,
             lvae_num_samples: int | None | UnsetType = UNSET,
             results: dict | None | UnsetType = UNSET,
             save_folder: Path | str | None | UnsetType = UNSET,
             overwrite: bool | UnsetType = UNSET,
             eval_gt: str | None | UnsetType = UNSET,
             data_prm_update: dict | None | UnsetType = UNSET,
             ch_out: int | None | UnsetType = UNSET,
             split: str | UnsetType = UNSET,
             limit_n_imgs: int | None | UnsetType = UNSET,
             timelapse_max: int | None | UnsetType = UNSET,
             evaluation_dataset_name: str | None = None,
             config: str | Path | None = None
             ):
    """Evaluate a saved run on a dataset split and optionally compute metrics.

    Any omitted optional argument is resolved from the local inference defaults
    or from the named config passed via `config`.
    """
    if evaluation_dataset_name is not None and split is not UNSET:
        raise ValueError("`--on` evaluates the complete evaluation dataset and cannot be combined with `--split`.")

    options = resolve_evaluate_options(
        config=config,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
        tiling_size=tiling_size,
        crop_size=crop_size,
        metrics_list=metrics_list,
        lvae_num_samples=lvae_num_samples,
        results=results,
        save_folder=save_folder,
        overwrite=overwrite,
        eval_gt=eval_gt,
        data_prm_update=data_prm_update,
        ch_out=ch_out,
        split=split,
        limit_n_imgs=limit_n_imgs,
        timelapse_max=timelapse_max,
    )
    
    run_dir = resolve_run_dir(dataset_name=dataset_name, subfolder=model_subfolder, exp_name=model_name)
    saved_run = load_saved_run(run_dir)
    evaluation_dataset = None
    if evaluation_dataset_name is not None:
        evaluation_dataset = resolve_evaluation_dataset(
            saved_run,
            evaluation_dataset_name,
            data_prm_update=options["data_prm_update"],
        )
        print(f"Evaluation dataset: {evaluation_dataset.name} (all data)")

    # Build list of checkpoints to evaluate (for example both best and last).
    run_options_list = _expand_checkpoint_selection(options)
    if len(run_options_list) > 1:
        print("best_or_last='both': running evaluation for checkpoints ['best', 'last'].")

    output_root = run_dir / "evaluations"
    for run_options in run_options_list:
        _run_single_evaluation(
            output_root=output_root,
            saved_run=saved_run,
            options=run_options,
            evaluation_dataset=evaluation_dataset,
        )
