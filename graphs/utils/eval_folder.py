from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from lisai.config import load_yaml
from lisai.evaluation import EvalSource
from lisai.runs.external import DiscoveredExternalRun, filter_external_runs, scan_external_runs

from graphs.utils.run_selection import (
    GraphRun,
    RunNames,
    discover_graph_runs,
)


def _extract_epoch(folder_name: str) -> int | None:
    match = re.search(
        r"(?:^|_)epoch_(\d+)(?:$|_)",
        folder_name,
    )

    if match is None:
        return None

    return int(match.group(1))


def get_eval_folder(
    root,
    evaluation_folder,
    ambiguity_selector="last_epoch",
):
    """
    Resolve an exact or epoch-suffixed folder below root.

    Kept for compatibility with existing graph scripts.
    """

    root = Path(root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Root folder does not exist or is not a directory: {root}"
        )

    exact_match = root / evaluation_folder

    if exact_match.is_dir():
        return exact_match

    selector = str(
        ambiguity_selector
    ).strip().lower()

    if selector == "exact":
        raise FileNotFoundError(
            f"No exact folder named '{evaluation_folder}' "
            f"found in '{root}'."
        )

    candidates = sorted(
        child
        for child in root.iterdir()
        if child.is_dir()
        and child.name.startswith(evaluation_folder)
    )

    if not candidates:
        raise FileNotFoundError(
            f"No evaluation folder found in '{root}' "
            f"matching '{evaluation_folder}' "
            f"or '{evaluation_folder}*'."
        )

    if len(candidates) == 1:
        return candidates[0].resolve()

    if selector in {
        "last_epoch",
        "first_epoch",
    }:
        candidates_with_epoch = []

        for candidate in candidates:
            epoch = _extract_epoch(candidate.name)

            if epoch is not None:
                candidates_with_epoch.append(
                    (epoch, candidate)
                )

        if not candidates_with_epoch:
            names = ", ".join(
                candidate.name
                for candidate in candidates
            )

            raise ValueError(
                "Ambiguous folders and no epoch number "
                f"was found: {names}"
            )

        reverse = selector == "last_epoch"

        candidates_with_epoch.sort(
            key=lambda item: item[0],
            reverse=reverse,
        )

        return candidates_with_epoch[0][1].resolve()

    raise ValueError(
        "ambiguity_selector must be "
        "'last_epoch', 'first_epoch', or 'exact'."
    )


def list_images(
    folder,
    selectors=(".tiff", ".tif"),
):
    """Legacy/general TIFF listing helper."""

    folder = Path(folder)

    if isinstance(selectors, str):
        selectors = (selectors,)

    suffixes = {
        suffix.lower()
        if suffix.startswith(".")
        else f".{suffix.lower()}"
        for suffix in selectors
    }

    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in suffixes
    )

    return files or None


@dataclass(frozen=True)
class ExternalGraphRun:
    discovered: DiscoveredExternalRun

    @property
    def run_dir(self) -> Path:
        return self.discovered.run_dir

    @property
    def name(self) -> str:
        return self.discovered.name

    @property
    def context_length(self) -> int:
        return 1

    @property
    def sampling_ratio(self) -> float | None:
        return None

    @property
    def beta_kl(self) -> float | None:
        return None


@dataclass(frozen=True)
class RunEvaluation:
    run: GraphRun | ExternalGraphRun
    folder: Path

    @property
    def run_dir(self) -> Path:
        return self.run.run_dir

    @property
    def name(self) -> str:
        return self.run.name

    @property
    def context_length(self) -> int:
        return self.run.context_length

    @property
    def sampling_ratio(self) -> float | None:
        return self.run.sampling_ratio

    @property
    def beta_kl(self) -> float | None:
        return self.run.beta_kl


def _validate_source_metadata(
    folder: Path,
    source: EvalSource,
) -> None:
    """
    Validate current-format evaluations.

    Older evaluations without evaluation.yaml are still accepted.
    """

    metadata_path = folder / "evaluation.yaml"

    if not metadata_path.is_file():
        return

    metadata = load_yaml(metadata_path)

    if not isinstance(metadata, dict):
        return

    dataset = metadata.get("dataset", {})

    if source.usage == "training":
        valid = (
            dataset.get("usage") == "training"
            and dataset.get("split") == source.name
        )

    else:
        valid = (
            dataset.get("usage") == "evaluation"
            and dataset.get("name") == source.name
        )

    if not valid:
        raise ValueError(
            f"Evaluation metadata in {metadata_path} "
            f"does not match {source}."
        )


def resolve_run_evaluation(
    run: GraphRun,
    *,
    source: EvalSource,
    checkpoint: Literal["best", "last"] | int = "best",
    ambiguity_selector: str = "last_epoch",
) -> RunEvaluation:
    """Resolve one evaluation folder for one run."""

    source_root = (
        run.run_dir
        / "evaluations"
        / source.folder_name
    )

    if isinstance(checkpoint, int):
        folder_prefix = f"epoch_{checkpoint}"
    else:
        folder_prefix = checkpoint

    folder = get_eval_folder(
        source_root,
        folder_prefix,
        ambiguity_selector,
    )

    _validate_source_metadata(
        folder,
        source,
    )

    return RunEvaluation(
        run=run,
        folder=folder,
    )


def resolve_evaluations(
    runs: Sequence[GraphRun],
    *,
    source: EvalSource,
    checkpoint: Literal["best", "last"] | int = "best",
    ambiguity_selector: str = "last_epoch",
    skip_missing: bool = False,
) -> list[RunEvaluation]:
    """Resolve the same evaluation source across several runs."""

    evaluations = []

    for run in runs:
        try:
            evaluation = resolve_run_evaluation(
                run,
                source=source,
                checkpoint=checkpoint,
                ambiguity_selector=ambiguity_selector,
            )

        except FileNotFoundError:
            if skip_missing:
                continue

            raise

        evaluations.append(evaluation)

    return evaluations


def discover_evaluations(
    *,
    dataset: str,
    model_subfolder: str,
    run_names: RunNames = "all",
    source: EvalSource,
    checkpoint: Literal["best", "last"] | int = "best",
    ambiguity_selector: str = "last_epoch",
    skip_missing: bool = False,
) -> list[RunEvaluation]:
    """
    Convenience helper used for graph scripts:
    discover runs, then resolve their evaluation folders.
    """

    runs = discover_graph_runs(
        dataset=dataset,
        model_subfolder=model_subfolder,
        run_names=run_names,
    )

    return resolve_evaluations(
        runs,
        source=source,
        checkpoint=checkpoint,
        ambiguity_selector=ambiguity_selector,
        skip_missing=skip_missing,
    )

def discover_external_evaluations(
    *,
    dataset: str,
    run_names: RunNames = "all",
    source: EvalSource,
    skip_missing: bool = False,
) -> list[RunEvaluation]:
    """Discover imported external runs and resolve their imported evaluation folders."""
    scan = scan_external_runs()
    candidates = filter_external_runs(scan.runs, dataset=dataset)
    if run_names == "all":
        selected = candidates
    else:
        by_name = {run.name: run for run in candidates}
        missing = [name for name in run_names if name not in by_name]
        if missing:
            available = ", ".join(sorted(by_name)) or "<none>"
            raise ValueError(
                f"Could not find requested external runs: {missing}. Available matching runs: {available}"
            )
        selected = [by_name[name] for name in run_names]

    evaluations: list[RunEvaluation] = []
    for discovered in selected:
        run = ExternalGraphRun(discovered)
        folder = run.run_dir / "evaluations" / source.folder_name / "imported"
        if not folder.is_dir():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Imported evaluation folder not found: {folder}")
        _validate_source_metadata(folder, source)
        evaluations.append(RunEvaluation(run=run, folder=folder))
    return evaluations
