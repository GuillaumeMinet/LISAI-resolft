from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from lisai.runs.cli import add_run_filter_arguments
from lisai.runs.scanner import DiscoveredRun
from lisai.runs.selection import resolve_discovered_run_selector

from .defaults import UNSET
from .run_apply_model import run_apply_model
from .run_evaluate import run_evaluate



def _parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list with at least one value.")
    return items


def _parse_crop_size(value: str) -> int | tuple[int, int]:
    items = _parse_csv_list(value)
    if len(items) == 1:
        return int(items[0])
    if len(items) == 2:
        return int(items[0]), int(items[1])
    raise argparse.ArgumentTypeError("crop_size must be 'N' or 'H,W'.")


def _parse_tiling_size(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"off", "none", "disable", "disabled"}:
        return "off"
    try:
        parsed = int(normalized)
    except ValueError:
        raise argparse.ArgumentTypeError("tiling_size must be a positive integer, 'auto', or 'off'.") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("tiling_size must be greater than 0.")
    return parsed


def _maybe_unset(value):
    return UNSET if value is None else value


def _add_model_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    action: str,
    allow_promoted_model: bool,
) -> argparse._ArgumentGroup:
    group = parser.add_argument_group("Model selection")
    group.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    group.add_argument("--run-id", help=f"Stable run identifier to {action}.")
    if allow_promoted_model:
        group.add_argument(
            "--model",
            help="Use a locally promoted model by public name instead of a training run.",
        )
    add_run_filter_arguments(group, include_identity=False, include_status=False)
    return group


def _add_config_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group("Configuration")
    group.add_argument(
        "-c",
        "--config",
        help=(
            "Inference config path, or a config name from configs/inference with or without "
            ".yml/.yaml. Local configs are looked up first. Defaults to local/defaults.yml; "
            "configs outside local/ are standalone and must be complete."
        ),
    )
    return group


def _add_checkpoint_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group("Checkpoint")
    group.add_argument("--epoch-number", "--epoch_number", dest="epoch_number", type=int)
    group.add_argument(
        "--best-or-last",
        "--best_or_last",
        dest="best_or_last",
        choices=["best", "last", "both"],
    )
    return group


def _add_inference_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group("Inference")
    group.add_argument("--crop-size", "--crop_size", dest="crop_size", type=_parse_crop_size)
    group.add_argument("--tiling-size", "--tiling_size", dest="tiling_size", type=_parse_tiling_size)
    group.add_argument("--no-tiling", dest="tiling_size", action="store_const", const="off")
    group.add_argument("--lvae-num-samples", "--lvae_num_samples", dest="lvae_num_samples", type=int)
    group.add_argument(
        "--progress-bar",
        dest="progress_bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the local/configured tqdm progress-bar preference for this command.",
    )
    return group


def add_apply_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    _add_model_selection_arguments(parser, action="apply", allow_promoted_model=True)
    _add_config_argument(parser)
    _add_checkpoint_arguments(parser)

    input_group = parser.add_argument_group("Input selection")
    input_group.add_argument("data_path", help="Input file or directory to process.")
    input_group.add_argument("--filters", type=_parse_csv_list)
    input_group.add_argument("--skip-if-contain", "--skip_if_contain", dest="skip_if_contain", type=_parse_csv_list)
    input_group.add_argument("--limit-n-imgs", "--limit_n_imgs", dest="limit_n_imgs", type=int)
    input_group.add_argument("--timelapse-max", "--timelapse_max", dest="timelapse_max", type=int)

    _add_inference_arguments(parser)

    output_location_group = parser.add_argument_group(
        "Output location",
        "CLI output options override inference-config and local-config saving preferences.",
    )
    output_choice = output_location_group.add_mutually_exclusive_group()
    output_choice.add_argument(
        "--save-folder",
        "--save_folder",
        dest="save_folder",
        metavar="PATH",
        help=(
            "Save predictions to PATH. If PATH already exists, LISAI creates a "
            "numbered sibling instead of overwriting it."
        ),
    )
    output_choice.add_argument(
        "--output-mode",
        "--output_mode",
        dest="output_mode",
        choices=["default", "in_place", "folder_inside", "folder_outside"],
        help=(
            "Choose prediction placement: default uses the configured inference "
            "directory; in_place writes directly with the input data; folder_inside creates "
            "a prediction folder inside the input directory; folder_outside creates it beside "
            "the input directory."
        ),
    )
    output_choice.add_argument(
        "--in-place",
        "--in_place",
        dest="in_place",
        action=argparse.BooleanOptionalAction,
        help=(
            "Save predictions alongside the input data. Use --no-in-place to "
            "explicitly force normal inference-directory routing."
        ),
    )

    output_contents_group = parser.add_argument_group("Output contents")
    output_contents_group.add_argument(
        "--save-input",
        dest="save_input",
        action=argparse.BooleanOptionalAction,
        help="Override the configured input-saving policy for this apply invocation.",
    )
    output_contents_group.add_argument(
        "--lvae-save-samples",
        "--lvae_save_samples",
        dest="lvae_save_samples",
        action=argparse.BooleanOptionalAction,
    )
    output_contents_group.add_argument(
        "--apply-color-code",
        "--apply_color_code",
        dest="apply_color_code",
        action=argparse.BooleanOptionalAction,
    )
    return parser


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    _add_model_selection_arguments(parser, action="evaluate", allow_promoted_model=False)
    _add_config_argument(parser)
    _add_checkpoint_arguments(parser)

    data_group = parser.add_argument_group("Evaluation data")
    data_group.add_argument(
        "--on",
        dest="evaluation_dataset_name",
        metavar="DATASET",
        help="Evaluate on the complete registered evaluation-only dataset instead of the run's own split.",
    )
    data_group.add_argument("--split")
    data_group.add_argument(
        "--eval-gt",
        "--eval_gt",
        dest="eval_gt",
        help="Evaluation GT path/key. Use @training for the saved training target or @none to disable GT.",
    )
    data_group.add_argument("--limit-n-imgs", "--limit_n_imgs", dest="limit_n_imgs", type=int)
    data_group.add_argument("--timelapse-max", "--timelapse_max", dest="timelapse_max", type=int)

    _add_inference_arguments(parser)

    metrics_group = parser.add_argument_group("Metrics")
    metrics_group.add_argument("--metrics", type=_parse_csv_list)

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--save-folder", "--save_folder", dest="save_folder")
    output_group.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    return parser


def run_apply_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    promoted_model_name = args.model
    if promoted_model_name is not None:
        if any((args.run, args.run_id, args.dataset, args.model_subfolder)):
            parser.error("--model cannot be combined with a run selector or run filters.")
        if args.epoch_number is not None or args.best_or_last is not None:
            parser.error("--epoch-number/--best-or-last do not apply to promoted models; promotion already fixes the checkpoint.")
        model_dataset = ""
        model_subfolder = "promoted"
        model_name = promoted_model_name
    else:
        selected = _resolve_run_selector(args)
        if selected is None:
            return 1
        model_dataset = selected.dataset
        model_subfolder = selected.model_subfolder
        model_name = selected.run_dir.name

    run_apply_model(
        model_dataset=model_dataset,
        model_subfolder=model_subfolder,
        model_name=model_name,
        data_path=Path(args.data_path),
        config=args.config,
        save_folder=_maybe_unset(args.save_folder),
        output_mode=_maybe_unset(args.output_mode),
        in_place=_maybe_unset(args.in_place),
        epoch_number=_maybe_unset(args.epoch_number),
        best_or_last=_maybe_unset(args.best_or_last),
        filters=_maybe_unset(args.filters),
        skip_if_contain=_maybe_unset(args.skip_if_contain),
        limit_n_imgs=_maybe_unset(args.limit_n_imgs),
        timelapse_max=_maybe_unset(args.timelapse_max),
        crop_size=_maybe_unset(args.crop_size),
        tiling_size=_maybe_unset(args.tiling_size),
        lvae_num_samples=_maybe_unset(args.lvae_num_samples),
        lvae_save_samples=_maybe_unset(args.lvae_save_samples),
        save_input=_maybe_unset(args.save_input),
        apply_color_code=_maybe_unset(args.apply_color_code),
        progress_bar=getattr(args, "progress_bar", None),
        promoted_model_name=promoted_model_name,
    )
    return 0


def run_evaluate_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.evaluation_dataset_name is not None and args.split is not None:
        parser.error("--on evaluates the complete evaluation dataset and cannot be combined with --split.")

    selected = _resolve_run_selector(args)
    if selected is None:
        return 1

    run_evaluate(
        dataset_name=selected.dataset,
        model_subfolder=selected.model_subfolder,
        model_name=selected.run_dir.name,
        config=args.config,
        best_or_last=_maybe_unset(args.best_or_last),
        epoch_number=_maybe_unset(args.epoch_number),
        tiling_size=_maybe_unset(args.tiling_size),
        crop_size=_maybe_unset(args.crop_size),
        metrics_list=_maybe_unset(args.metrics),
        lvae_num_samples=_maybe_unset(args.lvae_num_samples),
        save_folder=_maybe_unset(args.save_folder),
        overwrite=_maybe_unset(args.overwrite),
        eval_gt=_maybe_unset(args.eval_gt),
        split=_maybe_unset(args.split),
        limit_n_imgs=_maybe_unset(args.limit_n_imgs),
        timelapse_max=_maybe_unset(args.timelapse_max),
        evaluation_dataset_name=args.evaluation_dataset_name,
        progress_bar=getattr(args, "progress_bar", None),
    )
    return 0


def _resolve_run_selector(
    args: argparse.Namespace,
) -> DiscoveredRun | None:
    return resolve_discovered_run_selector(
        selector=args.run,
        run_id=args.run_id,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _apply_description() -> str:
    return (
        "Apply a trained model to raw data (noisy or subsampled)."
        "\n\nINFERENCE PARAMETERS: by default, apply uses the local default inference config "
        "configs/inference/local/defaults.yml. "
        "Common worfklow parameters such as 'lvae-num-samples' or 'tiling-size' are overridable " 
        "with CLI argument. For more advanced or model-specific inference settings, custom inference "
        "configurations can be passed by as CLI argument (--config)."

        "\n\nSAVING LOCATION AND BEHAVIOR: different saving modes are available. The 'default' mode saves "
        "inside the LISAI inference directory('inference_dir, configurable in local_config.yaml), while the "
        "the other 3 modes save outputs next to source data - see 'Output location' below for full detail. "
        "Input saving is also configurable in local_config.yaml with parameter 'save_input_mode'. "
        "A named config or a direct CLI option overrides any of the behavior. Additionnally, "
        "a specific saving folder can directly be specificied with --save-folder, see below."
    )


def _evaluate_description() -> str:
    return (
        "Evaluate a trained model on its dataset split or a registered evaluation dataset.\n\n"
        "\n\nEVALUATION PARAMETERS: by default, evaluate uses the local default inference config "
        "configs/inference/local/defaults.yml. "
        "Common worfklow parameters such as 'lvae-num-samples' or 'tiling-size' are overridable " 
        "with CLI argument. For more advanced or model-specific inference settings, custom inference "
        "configurations can be passed by as CLI argument (--config)."
    )


def add_apply_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "apply",
        help="Apply a trained model to one file or directory.",
        description=_apply_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_apply_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_apply_from_args(args, p))
    return parser


def add_evaluate_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model on its dataset split or a registered evaluation dataset.",
        description=_evaluate_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_evaluate_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_evaluate_from_args(args, p))
    return parser


def build_apply_parser(*, prog: str = "lisai apply") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=_apply_description(),
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_apply_arguments(parser)
    return parser


def build_evaluate_parser(*, prog: str = "lisai evaluate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=_evaluate_description(),
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_evaluate_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lisai evaluation")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    add_apply_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
