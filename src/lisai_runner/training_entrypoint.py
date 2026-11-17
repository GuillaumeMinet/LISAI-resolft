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
    parser.add_argument("--local-preferences", type=Path, help=argparse.SUPPRESS)
    return parser


def run_training(
    config: str | Path, *, workspace: str | Path | None = None,
    config_root: str | Path | None = None,
    local_preferences: str | Path | None = None,
) -> int:
    from lisai.api import train
    from lisai.config.settings import Settings, load_default_settings

    if workspace is not None:
        if config_root is None:
            raise ValueError("Runner child --workspace requires --config-root")
        settings = Settings.from_workspace(
            workspace, config_root=config_root, local_config=local_preferences,
        )
    elif config_root is not None or local_preferences is not None:
        raise ValueError("Runner child --config-root/--local-preferences requires --workspace")
    else:
        # Direct use (not a queued child) follows public API LISAI_HOME selection.
        settings = load_default_settings()
    train(Path(config), settings=settings)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_training(
        args.config, workspace=args.workspace, config_root=args.config_root,
        local_preferences=args.local_preferences,
    )


if __name__ == "__main__":
    raise SystemExit(main())
