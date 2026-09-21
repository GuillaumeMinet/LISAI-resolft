from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from lisai.config import ResolvedExperiment, load_yaml, settings
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.infra.paths import Paths
from lisai.runs import filter_runs, scan_runs
from lisai.runs.scanner import DiscoveredRun


RunNames = Sequence[str] | Literal["all"]


@dataclass(frozen=True)
class GraphRun:
    """Run information commonly needed by graph scripts."""

    discovered: DiscoveredRun
    saved: SavedTrainingRun
    config: ResolvedExperiment

    @property
    def run_dir(self) -> Path:
        return self.discovered.run_dir

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def context_length(self) -> int:
        # Single-frame runs may have no explicit timelapse context.
        return self.saved.context_length or 1

    @property
    def sampling_ratio(self) -> float | None:
        # Prefer the high-level task value when available.
        task_value = getattr(
            self.config.experiment.task,
            "sampling_ratio",
            None,
        )
        if task_value is not None:
            return float(task_value)

        # Multi-frame upsampling resolves this into the low-level
        # downsampling configuration.
        downsampling = self.config.data.downsampling
        if downsampling is None:
            return None

        if downsampling.downsamp_method == "random":
            if downsampling.downsamp_factor is None:
                return None
            return 1.0/downsampling.downsamp_factor**2

        if downsampling.multiple_prm is None:
            return None

        return float(downsampling.multiple_prm.fill_factor)

    @property
    def beta_kl(self) -> float | None:
        task_value = getattr(
            self.config.experiment.task,
            "betaKL",
            None,
        )
        if task_value is not None:
            return float(task_value)

        training_value = getattr(
            self.config.training,
            "betaKL",
            None,
        )
        if training_value is None:
            return None

        return float(training_value)

    @property
    def task_name(self) -> str:
        return str(self.config.experiment.task.name)


def load_graph_run(run: DiscoveredRun) -> GraphRun:
    """Load one discovered run and its resolved training configuration."""

    run_dir = run.run_dir

    cfg_path = Paths(settings).cfg_train_path(
        run_dir=run_dir,
    )

    config = ResolvedExperiment.model_validate(
        load_yaml(cfg_path)
    )

    saved = SavedTrainingRun.from_resolved(
        config,
        run_dir=run_dir,
    )

    return GraphRun(
        discovered=run,
        saved=saved,
        config=config,
    )


def discover_graph_runs(
    *,
    dataset: str,
    model_subfolder: str,
    run_names: RunNames = "all",
    status: str | None = None,
    kept: bool | None = None,
) -> list[GraphRun]:
    """
    Discover runs using LISAI's canonical run discovery.

    run_names="all":
        use all matching runs.

    run_names=[...]:
        use exact run-directory names and preserve the supplied order.
    """

    scan = scan_runs()

    candidates = filter_runs(
        scan.runs,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
        kept=kept,
    )

    if run_names == "all":
        selected = list(candidates)

    else:
        by_name = {
            run.run_dir.name: run
            for run in candidates
        }

        missing = [
            name
            for name in run_names
            if name not in by_name
        ]

        if missing:
            available = ", ".join(sorted(by_name)) or "<none>"

            raise ValueError(
                f"Could not find requested runs: {missing}. "
                f"Available matching runs: {available}"
            )

        selected = [
            by_name[name]
            for name in run_names
        ]

    return [
        load_graph_run(run)
        for run in selected
    ]