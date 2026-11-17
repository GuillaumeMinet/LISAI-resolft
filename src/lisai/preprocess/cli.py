from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from lisai.config import load_yaml
from lisai.config.settings import Settings, load_default_settings
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.infra.cli.prompts import is_interactive, prompt_yes_no
from lisai.infra.paths import Paths

from .reporting import ConsolePreprocessReporter, PreprocessReporter
from .run_preprocess import ExistingPreprocessOutput, PreprocessRun

class PreprocessAbortedError(RuntimeError):
    pass

def resolve_config_path(config_arg: str, *, resolver: ConfigPathResolver) -> Path:
    reference = resolver.resolve_reference(config_arg)
    assert reference is not None
    if reference.ownership == "built-in":
        raise ValueError(
            "Preprocess templates cannot be run directly. Create an editable config "
            f"with `lisai configs new preprocess {Path(config_arg).stem} --dataset YOUR_DATASET` "
            "and then run `lisai preprocess YOUR_DATASET`."
        )
    return Path(reference.location)

def _get_config_arg(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    positional = getattr(args, "config", None)
    option = getattr(args, "config_option", None)

    if positional and option:
        parser.error("Provide the preprocess config either positionally or via --config, not both.")
    if positional:
        return positional
    if option:
        return option

    parser.error("A preprocess config is required.")
    raise AssertionError("argparse.error should raise SystemExit")


def _confirm_overwrite(
    existing_output: ExistingPreprocessOutput,
    *,
    input_fn: Callable[[str], str],
) -> bool:
    prompt = (
        f"{existing_output.describe()}\n"
        "Overwriting will delete the current preprocess content and log. Continue? [y/N]: "
    )
    return bool(prompt_yes_no(prompt, input_fn=input_fn))


def run_preprocess_config(
    config_path: str | Path,
    *,
    paths: Paths,
    reporter: PreprocessReporter | None = None,
    input_fn: Callable[[str], str] = input,
    interactive: bool | None = None,
):
    cfg = load_yaml(config_path)
    if not isinstance(cfg, dict) or cfg.get("dataset_name") in (None, "", "CHANGEME"):
        raise ValueError("Preprocess dataset_name must be set to a real dataset, not CHANGEME")
    active_reporter = ConsolePreprocessReporter() if reporter is None else reporter

    run = PreprocessRun.from_cfg(cfg, paths=paths)
    existing_output = run.existing_output()

    overwrite = False
    if existing_output.exists:
        interactive_input = is_interactive(sys.stdin) if interactive is None else interactive
        if not interactive_input:
            raise FileExistsError(existing_output.describe())
        if not _confirm_overwrite(existing_output, input_fn=input_fn):
            raise PreprocessAbortedError("Preprocess aborted. Existing preprocess output was left untouched.")
        overwrite = True

    return run.execute(overwrite=overwrite, reporter=active_reporter)


def run_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, settings: Settings | None = None,
) -> int:
    config_arg = _get_config_arg(args, parser)
    stg = load_default_settings() if settings is None else settings
    resolver = ConfigPathResolver("preprocess", stg=stg)
    try:
        run_preprocess_config(resolve_config_path(config_arg, resolver=resolver), paths=Paths(stg))
    except (FileNotFoundError, ValueError, FileExistsError, PreprocessAbortedError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    return 0


def add_preprocess_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "config",
        nargs="?",
        help="Preprocess YAML file path or configuration name (with or without .yml/.yaml).",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help="Preprocess YAML file path or configuration name (with or without .yml/.yaml).",
    )
    return parser


def build_parser(*, prog: str = "lisai preprocess") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dataset preprocessing from a YAML config", prog=prog)
    add_preprocess_arguments(parser)
    return parser


def add_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "preprocess",
        help="Run dataset preprocessing from a YAML config.",
        description="Run dataset preprocessing from a YAML config",
    )
    add_preprocess_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_from_args(args, p, settings=getattr(args, "_lisai_settings", None)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_from_args(args, parser)
