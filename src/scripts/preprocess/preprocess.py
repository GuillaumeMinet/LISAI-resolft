from __future__ import annotations

import argparse

from lisai.preprocess.cli import main as cli_main
from lisai.preprocess.cli import run_preprocess_config


def main(cfg_path: str):
    return run_preprocess_config(cfg_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LISAI preprocessing from a YAML config.")
    parser.add_argument("config", nargs="?", help="Path to preprocess config YAML file.")
    parser.add_argument("-c", "--config", dest="config_option")
    args = parser.parse_args()

    argv = []
    if args.config:
        argv.append(args.config)
    if args.config_option:
        argv.extend(["--config", args.config_option])
    raise SystemExit(cli_main(argv))
