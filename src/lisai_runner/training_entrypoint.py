from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser(*, prog: str = "lisai-runner train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a LISAI training job through the public LISAI API.",
    )
    parser.add_argument("--config", required=True, help="Absolute path to training config snapshot.")
    parser.add_argument("--workspace", type=Path, help="Submitted workspace root (internal).")
    parser.add_argument("--config-root", type=Path, help="Submitted writable config root (internal).")
    parser.add_argument(
        "--settings-config-dir", type=Path, help=argparse.SUPPRESS,
    )
    return parser


def run_training(
    config: str | Path, *, workspace: str | Path | None = None,
    config_root: str | Path | None = None,
    settings_config_dir: str | Path | None = None,
) -> int:
    from lisai.api import train
    from lisai.config.settings import Settings, load_default_settings

    if settings_config_dir is not None:
        if workspace is not None or config_root is not None:
            raise ValueError("Legacy settings-config-dir cannot be combined with workspace/config-root")
        # Explicit backward compatibility for old jobs only; new jobs must use
        # the two-root workspace contract.
        settings = Settings.from_config_dir(settings_config_dir)
    elif workspace is not None:
        settings = Settings.from_workspace(workspace, config_root=config_root)
    elif config_root is not None:
        raise ValueError("Runner child --config-root requires --workspace")
    else:
        # Direct use (not a queued child) retains the public API default.
        settings = load_default_settings()
    train(Path(config), settings=settings)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_training(
        args.config, workspace=args.workspace, config_root=args.config_root,
        settings_config_dir=args.settings_config_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
