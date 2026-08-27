from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

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


def _parse_key_value_overrides(values: list[str] | None, parser: argparse.ArgumentParser) -> dict | object:
    if not values:
        return UNSET

    out: dict[str, object] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep:
            parser.error(f"Expected KEY=VALUE override, got: {value}")
        key = key.strip()
        if not key:
            parser.error(f"Override key cannot be empty: {value}")
        out[key] = yaml.safe_load(raw)
    return out


def _maybe_unset(value):
    return UNSET if value is None else value


def add_apply_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("data_path", help="Input file or directory to process.")
    parser.add_argument("--run-id", help="Stable run identifier to apply.")
    parser.add_argument("--model", help="Use a locally promoted model by public name instead of a training run.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.add_argument(
        "-c",
        "--config",
        help="Inference config path, or a config name from configs/inference with or without .yml/.yaml. Defaults to defaults.yml.",
    )
    parser.add_argument("--save-folder", "--save_folder", dest="save_folder")
    parser.add_argument("--in-place", "--in_place", dest="in_place", action=argparse.BooleanOptionalAction)
    parser.add_argument("--epoch-number", "--epoch_number", dest="epoch_number", type=int)
    parser.add_argument("--best-or-last", "--best_or_last", dest="best_or_last", choices=["best", "last", "both"])
    parser.add_argument("--filters", type=_parse_csv_list)
    parser.add_argument("--skip-if-contain", "--skip_if_contain", dest="skip_if_contain", type=_parse_csv_list)
    parser.add_argument("--crop-size", "--crop_size", dest="crop_size", type=_parse_crop_size)
    parser.add_argument(
        "--keep-original-shape",
        "--keep_original_shape",
        dest="keep_original_shape",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--tiling-size", "--tiling_size", dest="tiling_size", type=int)
    parser.add_argument("--stack-selection-idx", "--stack_selection_idx", dest="stack_selection_idx", type=int)
    parser.add_argument("--timelapse-max", "--timelapse_max", dest="timelapse_max", type=int)
    parser.add_argument("--lvae-num-samples", "--lvae_num_samples", dest="lvae_num_samples", type=int)
    parser.add_argument(
        "--lvae-save-samples",
        "--lvae_save_samples",
        dest="lvae_save_samples",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--denormalize-output",
        "--denormalize_output",
        dest="denormalize_output",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--save-inp", "--save_inp", dest="save_inp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--downsamp", type=int)
    parser.add_argument(
        "--apply-color-code",
        "--apply_color_code",
        dest="apply_color_code",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--color-code-option",
        "--color_code_option",
        dest="color_code_option",
        action="append",
        metavar="KEY=VALUE",
        help="Override nested apply.color_code_prm values, for example 'saturation=0.5'.",
    )
    parser.add_argument(
        "--dark-frame-context-length",
        "--dark_frame_context_length",
        dest="dark_frame_context_length",
        action=argparse.BooleanOptionalAction,
    )
    return parser


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("--run-id", help="Stable run identifier to evaluate.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.add_argument(
        "-c",
        "--config",
        help="Inference config path, or a config name from configs/inference with or without .yml/.yaml. Defaults to defaults.yml.",
    )
    parser.add_argument("--best-or-last", "--best_or_last", dest="best_or_last", choices=["best", "last", "both"])
    parser.add_argument("--epoch-number", "--epoch_number", dest="epoch_number", type=int)
    parser.add_argument("--tiling-size", "--tiling_size", dest="tiling_size", type=int)
    parser.add_argument("--crop-size", "--crop_size", dest="crop_size", type=_parse_crop_size)
    parser.add_argument("--metrics", type=_parse_csv_list)
    parser.add_argument("--lvae-num-samples", "--lvae_num_samples", dest="lvae_num_samples", type=int)
    parser.add_argument("--save-folder", "--save_folder", dest="save_folder")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--eval-gt",
        "--eval_gt",
        dest="eval_gt",
        help="Evaluation GT path/key. Use @training for the saved training target or @none to disable GT.",
    )
    parser.add_argument(
        "--data-option",
        "--data_option",
        dest="data_option",
        action="append",
        metavar="KEY=VALUE",
        help="Override nested evaluate.data_prm_update values, for example 'data_dir=/tmp/data'.",
    )
    parser.add_argument("--ch-out", "--ch_out", dest="ch_out", type=int)
    parser.add_argument("--split")
    parser.add_argument("--limit-n-imgs", "--limit_n_imgs", dest="limit_n_imgs", type=int)
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
        in_place=_maybe_unset(args.in_place),
        epoch_number=_maybe_unset(args.epoch_number),
        best_or_last=_maybe_unset(args.best_or_last),
        filters=_maybe_unset(args.filters),
        skip_if_contain=_maybe_unset(args.skip_if_contain),
        crop_size=_maybe_unset(args.crop_size),
        keep_original_shape=_maybe_unset(args.keep_original_shape),
        tiling_size=_maybe_unset(args.tiling_size),
        stack_selection_idx=_maybe_unset(args.stack_selection_idx),
        timelapse_max=_maybe_unset(args.timelapse_max),
        lvae_num_samples=_maybe_unset(args.lvae_num_samples),
        lvae_save_samples=_maybe_unset(args.lvae_save_samples),
        denormalize_output=_maybe_unset(args.denormalize_output),
        save_inp=_maybe_unset(args.save_inp),
        downsamp=_maybe_unset(args.downsamp),
        apply_color_code=_maybe_unset(args.apply_color_code),
        color_code_prm=_parse_key_value_overrides(args.color_code_option, parser),
        dark_frame_context_length=_maybe_unset(args.dark_frame_context_length),
        promoted_model_name=promoted_model_name,
    )
    return 0


def run_evaluate_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
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
        data_prm_update=_parse_key_value_overrides(args.data_option, parser),
        ch_out=_maybe_unset(args.ch_out),
        split=_maybe_unset(args.split),
        limit_n_imgs=_maybe_unset(args.limit_n_imgs),
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


def add_apply_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "apply",
        help="Apply a trained model to one file or directory.",
        description="Apply a trained model to image file(s)",
    )
    add_apply_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_apply_from_args(args, p))
    return parser


def add_evaluate_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model on a dataset split.",
        description="Evaluate a trained model on a dataset split",
    )
    add_evaluate_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_evaluate_from_args(args, p))
    return parser


def build_apply_parser(*, prog: str = "lisai apply") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a trained model to image file(s)", prog=prog)
    add_apply_arguments(parser)
    return parser


def build_evaluate_parser(*, prog: str = "lisai evaluate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split", prog=prog)
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
