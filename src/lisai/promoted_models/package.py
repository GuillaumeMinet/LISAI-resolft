from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lisai.config import load_yaml, save_yaml, settings
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.infra.paths import Paths

from .card import render_model_card, update_model_card_overview
from .checkpoint import extract_model_weights
from .promotion import PromotionPlan, build_promotion_plan
from .registry import load_promoted_model_registry, register_promoted_model, resolve_registered_model_dir
from .schema import PromotedModelManifest, PromotedModelRegistryEntry

MODEL_MANIFEST_FILENAME = "lisai_model.yaml"
MODEL_CARD_FILENAME = "README.md"


@dataclass(frozen=True)
class PromotedModel:
    model_dir: Path
    manifest: PromotedModelManifest

    @property
    def saved_run(self) -> SavedTrainingRun:
        config_path = self.model_dir / self.manifest.artifacts.config
        from lisai.config.models import ResolvedExperiment

        resolved_cfg = ResolvedExperiment.model_validate(load_yaml(config_path))
        return SavedTrainingRun.from_resolved(resolved_cfg, run_dir=self.model_dir)

    @property
    def weights_path(self) -> Path:
        return self.model_dir / self.manifest.artifacts.weights

    @property
    def noise_model_path(self) -> Path | None:
        if self.manifest.artifacts.noise_model is None:
            return None
        return self.model_dir / self.manifest.artifacts.noise_model.model

    @property
    def noise_model_norm_prm_path(self) -> Path | None:
        if self.manifest.artifacts.noise_model is None:
            return None
        return self.model_dir / self.manifest.artifacts.noise_model.norm_prm


@dataclass(frozen=True)
class PromotedModelExport:
    archive_path: Path
    archive_sha256: str
    manifest: PromotedModelManifest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_plan_files(plan: PromotionPlan, model_dir: Path) -> None:
    for item in plan.copy_files:
        destination = model_dir / item.package_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, destination)


def _payload_paths(manifest: PromotedModelManifest) -> list[str]:
    paths = [manifest.artifacts.config, manifest.artifacts.weights]
    if manifest.artifacts.loss is not None:
        paths.append(manifest.artifacts.loss)
    if manifest.artifacts.loss_plot is not None:
        paths.append(manifest.artifacts.loss_plot)
    if manifest.artifacts.noise_model is not None:
        paths.extend(
            [
                manifest.artifacts.noise_model.model,
                manifest.artifacts.noise_model.norm_prm,
            ]
        )
    return paths


def _write_manifest(model_dir: Path, manifest: PromotedModelManifest) -> None:
    save_yaml(
        manifest.model_dump(mode="json", exclude_none=True),
        model_dir / MODEL_MANIFEST_FILENAME,
    )


def load_promoted_model_from_dir(model_dir: str | Path) -> PromotedModel:
    model_dir = Path(model_dir)
    manifest_path = model_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Promoted-model manifest is missing: {manifest_path}")
    manifest = PromotedModelManifest.model_validate(load_yaml(manifest_path))
    required_paths = [
        model_dir / manifest.artifacts.config,
        model_dir / manifest.artifacts.weights,
    ]
    if manifest.artifacts.noise_model is not None:
        required_paths.extend(
            [
                model_dir / manifest.artifacts.noise_model.model,
                model_dir / manifest.artifacts.noise_model.norm_prm,
            ]
        )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Promoted model is incomplete; missing required artifact(s):\n"
            + "\n".join(str(path) for path in missing)
        )
    return PromotedModel(model_dir=model_dir, manifest=manifest)


def load_promoted_model(name: str, *, paths: Paths | None = None) -> PromotedModel:
    model_dir = resolve_registered_model_dir(name, paths=paths)
    promoted = load_promoted_model_from_dir(model_dir)
    if promoted.manifest.name != name:
        raise ValueError(
            f"Promoted-model registry name {name!r} disagrees with manifest name "
            f"{promoted.manifest.name!r}."
        )
    return promoted


def promote_run(
    run_dir: str | Path,
    *,
    name: str,
    checkpoint: str = "best",
    overwrite: bool = False,
    paths: Paths | None = None,
) -> PromotedModel:
    """Promote a training run into the canonical local reusable model library."""
    resolved_paths = paths or Paths(settings)
    plan = build_promotion_plan(
        run_dir,
        name=name,
        checkpoint=checkpoint,
        paths=resolved_paths,
    )
    model_dir = resolved_paths.promoted_model_dir(model_name=name)
    registry_root = resolved_paths.promoted_models_root()
    registry = load_promoted_model_registry(paths=resolved_paths)

    if name in registry.models and not overwrite:
        raise FileExistsError(
            f"Promoted model {name!r} is already registered. Use --overwrite to replace it."
        )
    if model_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Promoted-model directory already exists: {model_dir}. Use --overwrite to replace it."
        )

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"lisai-promote-{name}-", dir=model_dir.parent) as tmp:
        staging_dir = Path(tmp)
        _copy_plan_files(plan, staging_dir)
        extract_model_weights(
            plan.checkpoint_path,
            staging_dir / plan.manifest.artifacts.weights,
        )
        checksums = {
            package_path: sha256_file(staging_dir / package_path)
            for package_path in _payload_paths(plan.manifest)
        }
        manifest = plan.manifest.model_copy(update={"checksums": checksums})
        manifest = PromotedModelManifest.model_validate(manifest.model_dump(mode="python"))
        _write_manifest(staging_dir, manifest)
        (staging_dir / MODEL_CARD_FILENAME).write_text(
            render_model_card(plan, manifest),
            encoding="utf-8",
        )

        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(staging_dir), str(model_dir))

    relative_path = model_dir.relative_to(registry_root).as_posix()
    register_promoted_model(
        name=name,
        entry=PromotedModelRegistryEntry(
            source_run_id=manifest.source.run_id,
            path=relative_path,
            created_at=manifest.created_at,
            task=manifest.model.task,
        ),
        overwrite=overwrite,
        paths=resolved_paths,
    )
    return load_promoted_model_from_dir(model_dir)


def _write_zip(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
            archive.write(source, arcname=source.relative_to(source_dir).as_posix())


def export_promoted_model(
    name: str,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    paths: Paths | None = None,
) -> PromotedModelExport:
    """Export one canonical promoted model as a self-contained .lisai.zip archive."""
    resolved_paths = paths or Paths(settings)
    promoted = load_promoted_model(name, paths=resolved_paths)
    default_name = f"{name}.lisai.zip"
    if output is None:
        output_path = resolved_paths.promoted_model_exports_dir() / default_name
    else:
        candidate = Path(output)
        if candidate.exists() and candidate.is_dir():
            output_path = candidate / default_name
        elif candidate.suffix.lower() == ".zip":
            output_path = candidate
        else:
            output_path = candidate / default_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Promoted-model export already exists: {output_path}. Use --overwrite to replace it."
        )
    if output_path.exists():
        output_path.unlink()
    _write_zip(promoted.model_dir, output_path)
    return PromotedModelExport(
        archive_path=output_path,
        archive_sha256=sha256_file(output_path),
        manifest=promoted.manifest,
    )

def sync_promoted_model(
        name: str,
        *,
        paths: Paths | None = None,
) -> PromotedModel:
    """ Synchronizes derived metadata from lisai_model.yaml"""
    resolved_paths = paths or Paths(settings)
    promoted = load_promoted_model(name,paths=resolved_paths)

    # refresh model card (README.md)
    card_path = promoted.model_dir / MODEL_CARD_FILENAME
    if not card_path.is_file():
        raise FileNotFoundError(f"Promoted model {name}'s card is missing.")

    card_text = card_path.read_text(encoding="utf-8")
    updated_card = update_model_card_overview(card_text,promoted.manifest)

    # Refresh the lightweight registry metadata while preserving
    # registry-specific fields such as origin and installed_at.
    registry = load_promoted_model_registry(paths=resolved_paths)
    entry = registry.models[name]
    updated_entry = entry.model_copy(update={"task": promoted.manifest.model.task})
    updated_entry = PromotedModelRegistryEntry.model_validate(
        updated_entry.model_dump(mode="python")
    )

    card_path.write_text(updated_card, encoding="utf-8")

    register_promoted_model(
        name=name,
        entry=updated_entry,
        overwrite=True,
        paths=resolved_paths
    )

    return promoted


def set_promoted_model_task(
        name: str,
        task: str,
        *,
        paths:Paths | None = None,
) -> PromotedModel:
    """ Update the task of a promoted model and synchronize derived metadata
    so that registry.task and README multiframes are up to date."""
    resolved_paths=paths or Paths(settings)
    promoted = load_promoted_model(name,paths=resolved_paths)

    updated_model = promoted.manifest.model.model_copy(
        update= {"task": task}
    )
    updated_manifest = promoted.manifest.model_copy(
        update={"model": updated_model}
    )
    updated_manifest = PromotedModelManifest.model_validate(
        updated_manifest.model_dump(mode="python")
    )

    # before commiting new manifest, check that model card (README.md)
    # accepts the new manifest by doing a card update dry-run.
    # This catches a badly formed manifest early.
    card_path = promoted.model_dir / MODEL_CARD_FILENAME
    if not card_path.is_file():
        raise FileNotFoundError(f"Promoted-model card is missing: {card_path}")
    card_text = card_path.read_text(encoding="utf-8")
    update_model_card_overview(card_text,updated_manifest) # ignore return, dry-run

    # now commit manifest
    _write_manifest(promoted.model_dir,updated_manifest)

    # sync derived metadata
    sync = sync_promoted_model(name,paths=resolved_paths)

    return sync

__all__ = [
    "MODEL_CARD_FILENAME",
    "MODEL_MANIFEST_FILENAME",
    "PromotedModel",
    "PromotedModelExport",
    "export_promoted_model",
    "load_promoted_model",
    "load_promoted_model_from_dir",
    "promote_run",
    "set_promoted_model_task",
    "sync_promoted_model",
    "sha256_file",
]
