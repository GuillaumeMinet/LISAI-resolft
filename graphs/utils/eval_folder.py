from __future__ import annotations

import re
from pathlib import Path


def _extract_epoch(folder_name: str) -> int | None:
    match = re.search(r"(?:^|_)epoch_(\d+)(?:$|_)", folder_name)
    if match is None:
        return None
    return int(match.group(1))


def get_eval_folder(root, evaluation_folder, ambiguity_selector):
    """Resolve an evaluation folder under ``root``.

    Resolution order:
    1. Exact match ``root / evaluation_folder`` if it exists.
    2. If no exact match, search direct subfolders whose name starts with
       ``evaluation_folder``.
    3. If multiple candidates remain, resolve ambiguity with:
       - ``last_epoch``: highest epoch number
       - ``first_epoch``: lowest epoch number
       - ``exact``: fail unless an exact match exists
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root folder does not exist or is not a directory: {root}")

    exact_match = root / evaluation_folder
    if exact_match.is_dir():
        return exact_match

    selector = str(ambiguity_selector).strip().lower()
    if selector == "exact":
        raise FileNotFoundError(
            f"No exact folder named '{evaluation_folder}' found in '{root}'."
        )

    candidates = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith(evaluation_folder)
        ),
        key=lambda path: path.name,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No evaluation folder found in '{root}' matching '{evaluation_folder}' "
            f"or prefix '{evaluation_folder}*'."
        )

    if len(candidates) == 1:
        return candidates[0].resolve()

    if selector in {"last_epoch", "first_epoch"}:
        candidates_with_epoch = []
        for candidate in candidates:
            epoch = _extract_epoch(candidate.name)
            if epoch is not None:
                candidates_with_epoch.append((epoch, candidate))

        if not candidates_with_epoch:
            names = ", ".join(candidate.name for candidate in candidates)
            raise ValueError(
                "Ambiguous evaluation folders found, but none includes an epoch number "
                f"for selector '{ambiguity_selector}': {names}"
            )

        reverse = selector == "last_epoch"
        candidates_with_epoch.sort(key=lambda item: item[0], reverse=reverse)
        return candidates_with_epoch[0][1].resolve()

    names = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(
        "Ambiguous evaluation folders found. Use ambiguity_selector='last_epoch', "
        "'first_epoch', or 'exact'. "
        f"Candidates: {names}"
    )


def list_images(folder, selectors=(".tiff", ".tif")):
    folder_path = Path(folder)
    if isinstance(selectors, str):
        selectors = (selectors,)

    normalized_selectors = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in selectors
    }

    final_list = []
    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in normalized_selectors:
            continue
        final_list.append(file_path)

    if not final_list:
        return None
    return final_list
