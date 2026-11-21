"""Workspace-bound catalogs for existing LISAI datasets, models, and runs."""
from __future__ import annotations

def get_installed_model_card(settings, model_name: str) -> str | None:
    """Read the local card only; never fetch model weights for a preview."""
    from lisai.infra.paths import Paths
    from lisai.promoted_models.package import MODEL_CARD_FILENAME, load_promoted_model

    promoted = load_promoted_model(model_name, paths=Paths(settings))
    card = promoted.model_dir / MODEL_CARD_FILENAME
    return card.read_text(encoding="utf-8") if card.is_file() else None


def list_reusable_training_datasets(settings) -> list[dict]:
    """List registered, unprepared TIFF datasets usable by quick HDN.

    No data are imported or copied. Split feasibility and noise-model checks
    remain authoritative in the L2 preparation API.
    """
    from lisai.data.dataset_registry import load_dataset_registry
    from lisai.infra.paths import Paths

    paths = Paths(settings)
    result = []
    for name, info in sorted(load_dataset_registry(paths.dataset_registry_path()).items()):
        if not isinstance(info, dict) or info.get("usage") != "training" or info.get("for_training") is False:
            continue
        fmt = info.get("data_format")
        if fmt not in {"single", "timelapse"}:
            continue
        folder = paths.dataset_dir(dataset_name=name, usage="training") / "raw"
        if not folder.is_dir():
            continue
        recordings = [file for file in folder.iterdir()
                      if file.is_file() and file.suffix in {".tif", ".tiff"}]
        if len(recordings) < 2:
            continue
        result.append({"name": name, "data_format": fmt, "recording_count": len(recordings)})
    return result


def list_usable_training_runs(settings) -> list[dict]:
    """Discover completed runs with checkpoint files; never promote them."""
    from lisai.evaluation.saved_run import load_saved_run
    from lisai.infra.paths import Paths
    from lisai.infra.paths.checkpoint_resolution import resolve_checkpoint_path
    from lisai.runs.scanner import scan_runs

    paths = Paths(settings)
    entries = []
    for run in scan_runs(paths=paths).runs:
        if run.metadata.status != "completed" or not run.path_consistent:
            continue
        try:
            saved = load_saved_run(run.run_dir, paths=paths)
            checkpoints = {}
            for selector in ("best", "last"):
                try:
                    resolve_checkpoint_path(
                        paths=paths, run_dir=run.run_dir,
                        load_methods=saved.checkpoint_methods, best_or_last=selector,
                    )
                    checkpoints[selector] = True
                except FileNotFoundError:
                    checkpoints[selector] = False
            if not any(checkpoints.values()):
                continue
        except (ValueError, OSError, FileNotFoundError):
            continue
        entries.append({
            "run_id": run.metadata.run_id,
            "name": run.metadata.run_name,
            "dataset": run.dataset,
            "run_dir": run.run_dir,
            "checkpoints": checkpoints,
            "label": f"{run.metadata.run_name}  [{run.dataset}]  ({run.metadata.run_id[:8]})",
        })
    return entries


def find_new_completed_run(settings, *, dataset_name: str, experiment_name: str,
                           old_run_ids: set[str]) -> str | None:
    """Use the run scanner's real IDs after subprocess completion, not guessed paths."""
    matches = [run for run in list_usable_training_runs(settings)
               if run["run_id"] not in old_run_ids and run["dataset"] == dataset_name
               and run["name"] == experiment_name]
    return matches[0]["run_id"] if len(matches) == 1 else None


def list_hdn_noise_models(settings) -> list[str]:
    """List locally available calibration directories; L2 validates contents."""
    from lisai.infra.paths import Paths

    paths = Paths(settings)
    root = paths.noise_model_dir(noiseModel_name="_lisai_probe_").parent
    if not root.is_dir():
        return []
    return sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and
        paths.noise_model_path(noiseModel_name=child.name).is_file() and
        paths.noise_model_norm_prm_path(noiseModel_name=child.name).is_file()
    )


def get_installed_model_names(settings) -> list[str]:
    """Look up only models installed into this Settings' workspace."""
    from lisai.infra.paths import Paths
    from lisai.promoted_models.registry import load_promoted_model_registry

    return sorted(load_promoted_model_registry(paths=Paths(settings)).models)
