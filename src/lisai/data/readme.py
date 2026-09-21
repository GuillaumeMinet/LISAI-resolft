from __future__ import annotations

from pathlib import Path

DATASET_README_FILENAME = "README.md"
DATASET_README_TEMPLATE = "README not updated yet.\n"


def dataset_readme_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / DATASET_README_FILENAME


def ensure_dataset_readme(dataset_dir: str | Path) -> tuple[Path, bool]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    readme_path = dataset_readme_path(dataset_path)
    if readme_path.exists():
        return readme_path, False

    readme_path.write_text(DATASET_README_TEMPLATE, encoding="utf-8")
    return readme_path, True
