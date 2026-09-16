from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lisai.config import load_yaml, settings
from lisai.config.models import ResolvedExperiment
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.infra.paths import Paths
from lisai.runs import RunMetadata, read_run_metadata

from .checkpoint import PromotionCheckpointSelector, resolve_promotion_checkpoint
from .dependencies import PromotionDependencies, PromotionFile, collect_required_dependencies
from .inference_config import (
    PROMOTED_INFERENCE_CONFIG_FILENAME,
    resolve_promoted_inference_config_source,
)
from .schema import (
    PromotedModelArtifacts,
    PromotedModelCodeState,
    PromotedModelDefinition,
    PromotedModelManifest,
    PromotedModelSource,
    PromotedTrainingData,
    PromotedTrainingSummary,
)

PROMOTABLE_RUN_STATUSES = frozenset({"completed", "stopped"})


@dataclass(frozen=True)
class PromotionPlan:
    """Validated, filesystem-aware plan for turning one run into a promoted model."""

    run_dir: Path
    metadata: RunMetadata
    saved_run: SavedTrainingRun
    manifest: PromotedModelManifest
    checkpoint_path: Path
    checkpoint_selector: PromotionCheckpointSelector
    copy_files: tuple[PromotionFile, ...]
    dependencies: PromotionDependencies


def _validate_promotable_status(metadata: RunMetadata) -> None:
    if metadata.status not in PROMOTABLE_RUN_STATUSES:
        allowed = ", ".join(sorted(PROMOTABLE_RUN_STATUSES))
        raise ValueError(
            f"Run {metadata.run_id} has status {metadata.status!r}. "
            f"Only {allowed} runs can be promoted."
        )


def _code_state(metadata: RunMetadata) -> PromotedModelCodeState | None:
    if metadata.code is None:
        return None
    return PromotedModelCodeState.model_validate(metadata.code.model_dump(mode="json"))


def _optional_run_artifact(source_path: Path, package_path: str) -> PromotionFile | None:
    if not source_path.exists():
        return None
    return PromotionFile(source_path=source_path, package_path=package_path)


def build_promotion_plan(
    run_dir: str | Path,
    *,
    name: str,
    checkpoint: PromotionCheckpointSelector = "best",
    inference_config: str | Path | None = None,
    paths: Paths | None = None,
) -> PromotionPlan:
    """Validate a saved run and project it into the promoted-model domain.

    This function does not mutate the run and does not create package files. It
    resolves all source artifacts up front so a later packaging step can be a
    straightforward execution of this plan.
    """
    run_dir = Path(run_dir)
    resolved_paths = paths or Paths(settings)
    metadata = read_run_metadata(run_dir)
    _validate_promotable_status(metadata)

    config_path = resolved_paths.cfg_train_path(run_dir=run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Saved training config is missing: {config_path}")
    resolved_cfg = ResolvedExperiment.model_validate(load_yaml(config_path))
    saved_run = SavedTrainingRun.from_resolved(resolved_cfg, run_dir=run_dir)

    if metadata.dataset != saved_run.dataset_name:
        raise ValueError(
            "Run metadata and saved training config disagree on dataset name: "
            f"{metadata.dataset!r} != {saved_run.dataset_name!r}."
        )

    checkpoint_path = resolve_promotion_checkpoint(
        saved_run,
        selector=checkpoint,
        paths=resolved_paths,
    )
    dependencies = collect_required_dependencies(saved_run, paths=resolved_paths)

    loss_file = _optional_run_artifact(
        resolved_paths.loss_file_path(run_dir=run_dir),
        "training/loss.txt",
    )
    loss_plot = _optional_run_artifact(
        resolved_paths.loss_plot_path(run_dir=run_dir),
        "training/loss_plot.png",
    )

    copy_files: list[PromotionFile] = [
        PromotionFile(config_path, "config_train.yaml"),
    ]
    inference_config_path: Path | None = None
    if inference_config is not None:
        inference_config_path = resolve_promoted_inference_config_source(inference_config)
        copy_files.append(
            PromotionFile(inference_config_path, PROMOTED_INFERENCE_CONFIG_FILENAME)
        )
    if loss_file is not None:
        copy_files.append(loss_file)
    if loss_plot is not None:
        copy_files.append(loss_plot)
    copy_files.extend(dependencies.files)

    task_name = str(resolved_cfg.experiment.task.name)
    architecture = saved_run.model_architecture
    training_signature = (
        metadata.training_signature.model_dump(mode="json")
        if metadata.training_signature is not None
        else None
    )
    runtime_stats = (
        metadata.runtime_stats.model_dump(mode="json")
        if metadata.runtime_stats is not None
        else None
    )

    manifest = PromotedModelManifest(
        name=name,
        model=PromotedModelDefinition(task=task_name, architecture=architecture),
        training_data=PromotedTrainingData(dataset=metadata.dataset),
        source=PromotedModelSource(
            run_id=metadata.run_id,
            run_name=metadata.run_name,
            run_status=metadata.status,
            checkpoint_selector=checkpoint,
            checkpoint_filename=checkpoint_path.name,
            code=_code_state(metadata),
        ),
        training=PromotedTrainingSummary(
            best_val_loss=metadata.best_val_loss,
            last_epoch=metadata.last_epoch,
            max_epoch=metadata.max_epoch,
            training_signature=training_signature,
            runtime_stats=runtime_stats,
        ),
        artifacts=PromotedModelArtifacts(
            inference_config=(
                PROMOTED_INFERENCE_CONFIG_FILENAME
                if inference_config_path is not None
                else None
            ),
            loss=loss_file.package_path if loss_file is not None else None,
            loss_plot=loss_plot.package_path if loss_plot is not None else None,
            noise_model=dependencies.noise_model,
        ),
    )

    return PromotionPlan(
        run_dir=run_dir,
        metadata=metadata,
        saved_run=saved_run,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_selector=checkpoint,
        copy_files=tuple(copy_files),
        dependencies=dependencies,
    )


__all__ = [
    "PROMOTABLE_RUN_STATUSES",
    "PromotionPlan",
    "build_promotion_plan",
]
