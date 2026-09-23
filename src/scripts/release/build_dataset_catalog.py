"""Build and validate the staged paper dataset catalog.

Example:
    python src/scripts/release/build_dataset_catalog.py \
        E:\\lisai E:\\lisai\\paper_dataset_pre_upload
"""

from __future__ import annotations

import argparse

from lisai.data.release_builder import ReleaseValidationError, build_dataset_catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate locally staged LISAI dataset ZIPs and generate dataset_catalog.json. "
            "The command is local-only: it does not create archives or contact Zenodo."
        )
    )
    parser.add_argument(
        "data_root",
        help="LISAI data root containing datasets/, noise_models/, etc.",
    )
    parser.add_argument(
        "staging_dir",
        help="Directory containing the ZIP files prepared for upload.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Catalog output path (default: <staging_dir>/dataset_catalog.json).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        build_dataset_catalog(
            args.data_root,
            args.staging_dir,
            output_path=args.output,
            progress=True,
        )
    except ReleaseValidationError as exc:
        print("\nRelease validation FAILED:")
        for error in exc.errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
