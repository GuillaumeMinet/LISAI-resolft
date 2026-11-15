from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser(*, prog: str = "lisai-runner train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a LISAI training job through the public LISAI API.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the resolved training config snapshot.",
    )
    parser.add_argument(
        "--settings-config-dir",
        type=Path,
        help="Legacy configuration directory chosen by the submitting queue (internal).",
    )
    return parser


def run_training(config: str | Path, *, settings_config_dir: str | Path | None = None) -> int:
    from lisai.api import train
    from lisai.config.settings import Settings, load_default_settings

    settings = (
        Settings.from_config_dir(settings_config_dir)
        if settings_config_dir is not None
        else load_default_settings()
    )
    train(Path(config), settings=settings)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_training(args.config, settings_config_dir=args.settings_config_dir)


if __name__ == "__main__":
    raise SystemExit(main())
