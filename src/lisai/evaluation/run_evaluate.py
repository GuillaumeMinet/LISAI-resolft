"""High-level entrypoint for dataset-based evaluation of a saved model.

This module orchestrates the full evaluation flow: resolve the saved run,
initialize the inference runtime, rebuild the evaluation sample source,
run predictions, compute metrics, and persist outputs.
"""

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

from lisai.evaluation.defaults import UNSET, UnsetType, resolve_evaluate_options
from lisai.evaluation.data import build_eval_source
from lisai.evaluation.inference.stack import infer_batch
from lisai.evaluation.io import (
    EvalItemOutputWriter,
    create_save_folder,
    ensure_save_folder,
    save_metrics_json,
)
from lisai.evaluation.metrics import compute as metrics
from lisai.evaluation.runtime import initialize_runtime
from lisai.evaluation.saved_run import SavedTrainingRun, load_saved_run, resolve_run_dir


def _build_evaluation_folder_name(
    *,
    best_or_last: str,
    requested_epoch: int | None,
    resolved_epoch: int | None,
    split: str,
) -> str:
    """Build the default output folder name for one evaluation run."""
    if requested_epoch is not None:
        save_name = f"evaluation_epoch_{requested_epoch}"
    elif resolved_epoch is not None:
        save_name = f"evaluation_{best_or_last}_epoch_{resolved_epoch}"
    else:
        save_name = f"evaluation_{best_or_last}"

    if split != "test":
        save_name = f"{save_name}_{split}"
    return save_name


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


def _run_single_evaluation(*, run_dir: Path, saved_run: SavedTrainingRun, options: dict[str, Any]) -> None:
    """Run one resolved checkpoint evaluation and save outputs/metrics."""
    runtime = initialize_runtime(
        saved_run=saved_run,
        best_or_last=options["best_or_last"],
        epoch_number=options["epoch_number"],
        tiling_size=options["tiling_size"],
    )

    if options["save_folder"] is None:
        save_name = _build_evaluation_folder_name(
            best_or_last=options["best_or_last"],
            requested_epoch=options["epoch_number"],
            resolved_epoch=runtime.resolved_epoch,
            split=options["split"],
        )
        save_folder = create_save_folder(path=run_dir / save_name,
                                         overwrite=options["overwrite"], parent_exists_check=True)
    else:
        save_folder = ensure_save_folder(Path(options["save_folder"]))

    if save_folder is None:
        raise FileNotFoundError("Model folder not found.")

    if saved_run.is_lvae:
        assert options["lvae_num_samples"] is not None, (
            "for LVAE prediction, number of samples needs to be specified"
        )

    upsamp = saved_run.upsampling_factor
    print(f"Found upsampling factor to be: {upsamp}\n")
    tiling_size = runtime.tiling_size
    print(f"Tiling size: {tiling_size}\n")

    sample_source = build_eval_source(
        saved_run,
        split=options["split"],
        crop_size=options["crop_size"],
        eval_gt=options["eval_gt"],
        data_prm_update=options["data_prm_update"],
    )
    print(f"Evaluation GT: {_format_eval_gt_for_display(sample_source.config.target)}")
    results = options["results"]

    n_processed = 0
    stop_eval = False
    for item_id, item in enumerate(sample_source.iter_items()):
        print(f"Item {item_id} / {len(sample_source.items)}: {item.name}")
        writer = EvalItemOutputWriter(item=item, save_folder=save_folder)

        for sample_index, sample in item.iter_samples(sample_source.config):
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
             tiling_size: int | None | UnsetType = UNSET,
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
             config: str | Path | None = None
             ):
    """Evaluate a saved run on a dataset split and optionally compute metrics.

    Any omitted optional argument is resolved from `configs/inference/defaults.yml`
    or from the named config passed via `config`.
    """
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
    )
    run_dir = resolve_run_dir(dataset_name=dataset_name, subfolder=model_subfolder, exp_name=model_name)
    saved_run = load_saved_run(run_dir)

    # build list of runs to evaluate (so that we can evaluate both best and last run for example)
    run_options_list = _expand_checkpoint_selection(options)
    if len(run_options_list) > 1:
        print("best_or_last='both': running evaluation for checkpoints ['best', 'last'].")

    for run_options in run_options_list:
        _run_single_evaluation(run_dir=run_dir, saved_run=saved_run, options=run_options)
