"""Toolkit-independent training preflight and subprocess invocation."""
from __future__ import annotations

import sys
from pathlib import Path

from .config_editor import managed_training_config_path, validate_training_yaml_text


class SavedTrainingPlan:
    """Launch parameters for a validated, user-authored training YAML."""

    def __init__(self, config_path: Path, dataset_name: str, experiment_name: str,
                 models_subfolder: str = "HDN", *, mode: str = "train", run_dir: Path | None = None):
        self.config_path = Path(config_path)
        self.dataset_name = str(dataset_name)
        self.experiment_name = str(experiment_name)
        self.models_subfolder = str(models_subfolder)
        self.mode = str(mode)
        self.run_dir = Path(run_dir) if run_dir is not None else None


def training_command(settings, plan, *, interpreter: str | None = None) -> list[str]:
    """Build an isolated LISAI worker invocation; never consult CLI selection."""
    executable = interpreter or sys.executable
    config_path = Path(plan.config_path).resolve()
    workspace = Path(settings.WORKSPACE_ROOT).resolve()
    config_root = Path(settings.CONFIGS_ROOT).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    if not config_root.is_dir() or not workspace.is_dir():
        raise ValueError("LISAI workspace or writable config directory is missing.")
    return [
        str(executable), "-u", "-m", "lisai_runner.training_entrypoint",
        "--config", str(config_path),
        "--workspace", str(workspace),
        "--config-root", str(config_root),
    ]


# Compatibility with the R1 Quick HDN public helper.
quick_hdn_training_command = training_command


def plan_for_training_config(config_path: str | Path, *, settings) -> SavedTrainingPlan:
    """Preflight any supported saved training YAML for the regular LISAI runner.

    Unlike Quick HDN preparation, this path does not assume unsupervised LVAE,
    unprepared data, a noise model, or file-level splitting. The resolved
    training engine remains authoritative for model/dataset compatibility.
    Refuse destructive overwrites in this GUI entry point.
    """
    from lisai.infra.paths import Paths

    path = managed_training_config_path(settings, config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    resolved = validate_training_yaml_text(path.read_text(encoding="utf-8"), settings)
    mode = resolved.experiment.mode
    if mode not in {"train", "retrain", "continue_training"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if mode != "continue_training" and resolved.experiment.overwrite:
        raise ValueError("GUI training refuses overwrite=true; use a new experiment name or the CLI.")
    dataset_name = resolved.data.dataset_name
    experiment_name = resolved.experiment.exp_name
    if not dataset_name or not experiment_name:
        raise ValueError("Training config must resolve a dataset and experiment name.")
    if mode == "continue_training":
        origin = resolved.experiment.origin_run_dir
        if not origin:
            raise ValueError("Continue training requires an origin run directory.")
        run_dir = Path(origin).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Original run does not exist: {run_dir}")
    else:
        run_dir = Paths(settings).run_dir(
            dataset_name=dataset_name, models_subfolder=resolved.routing.models_subfolder,
            exp_name=experiment_name,
        )
        if run_dir.exists():
            raise FileExistsError(f"Run already exists at {run_dir}; choose a different experiment name.")
    return SavedTrainingPlan(
        path, dataset_name, experiment_name, resolved.routing.models_subfolder,
        mode=mode, run_dir=run_dir,
    )


def plan_for_saved_training_config(config_path: str | Path, *, settings) -> SavedTrainingPlan:
    """Preflight a *fresh* quick HDN training run from a saved YAML."""
    from lisai.config.models.training import DataSection
    from lisai.data.data_loaders.split_manifest import collect_input_files, make_unprepared_split_manifest
    from lisai.infra.paths import Paths
    from lisai.training.quick_hdn import _check_noise_model

    path = managed_training_config_path(settings, config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    resolved = validate_training_yaml_text(path.read_text(encoding="utf-8"), settings)
    if resolved.experiment.mode != "train" or bool(resolved.experiment.overwrite):
        raise ValueError("Quick HDN launch supports fresh training configs with mode=train and overwrite=false.")
    if (resolved.model is None or resolved.model.architecture != "lvae" or
            resolved.data.prep_before or resolved.data.paired or resolved.data.already_split):
        raise ValueError("This quick HDN launcher requires unsupervised LVAE with prep_before=false and file-level splits.")
    name = resolved.data.dataset_name
    experiment_name = resolved.experiment.exp_name
    if not name or not experiment_name:
        raise ValueError("Training config must define data.dataset_name and experiment.exp_name.")
    paths = Paths(settings)
    run_dir = paths.run_dir(dataset_name=name,
                            models_subfolder=resolved.routing.models_subfolder,
                            exp_name=experiment_name)
    if run_dir.exists():
        raise FileExistsError(f"A run already exists at {run_dir}. Save As with a new experiment name.")
    noise_model_name = getattr(resolved.noise_model, "name", None)
    if not noise_model_name:
        raise ValueError("The HDN configuration must reference a calibrated noise model.")
    _check_noise_model(paths, noise_model_name)
    data_dir = paths.dataset_dir(dataset_name=name,
                                 data_subfolder=resolved.routing.data_subfolder,
                                 usage="training")
    data = DataSection.model_validate({**resolved.data.model_dump(exclude_none=True),
                                       "data_dir": data_dir})
    inputs = collect_input_files(data)
    if len(inputs) < 2:
        raise ValueError("At least two independent TIFF recordings are required.")
    make_unprepared_split_manifest(data, data_format=resolved.data.data_format)
    return SavedTrainingPlan(path, name, experiment_name, resolved.routing.models_subfolder)
