"""Small, toolkit-independent quick training launch contract and preflight."""
from __future__ import annotations

import sys
from pathlib import Path

from .config_editor import managed_training_config_path, validate_training_yaml_text


class SavedTrainingPlan:
    """Launch parameters for a validated, user-authored training YAML."""

    def __init__(self, config_path: Path, dataset_name: str, experiment_name: str,
                 models_subfolder: str = "HDN"):
        self.config_path = Path(config_path)
        self.dataset_name = str(dataset_name)
        self.experiment_name = str(experiment_name)
        self.models_subfolder = str(models_subfolder)


def quick_hdn_training_command(settings, plan, *, interpreter: str | None = None) -> list[str]:
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
