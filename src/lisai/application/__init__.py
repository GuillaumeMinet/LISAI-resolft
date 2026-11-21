"""Reusable, toolkit-independent application services.

Importing this module does not import Qt or ImProcess. Computation stays in
existing LISAI backend APIs; these helpers provide discovery and orchestration.
"""

from .catalogs import (
    find_new_completed_run,
    get_installed_model_card,
    get_installed_model_names,
    list_hdn_noise_models,
    list_reusable_training_datasets,
    list_usable_training_runs,
)
from .config_editor import (
    list_user_training_configs,
    managed_training_config_path,
    save_training_yaml_text,
    validate_training_yaml_text,
)
from .training import (
    SavedTrainingPlan,
    plan_for_saved_training_config,
    plan_for_training_config,
    training_command,
    quick_hdn_training_command,
)
from .workspace import create_workspace, validate_workspace

__all__ = [
    "create_workspace", "validate_workspace",
    "find_new_completed_run", "get_installed_model_card", "get_installed_model_names",
    "list_hdn_noise_models", "list_reusable_training_datasets", "list_usable_training_runs",
    "list_user_training_configs", "managed_training_config_path", "save_training_yaml_text",
    "validate_training_yaml_text", "SavedTrainingPlan", "plan_for_saved_training_config",
    "quick_hdn_training_command", "plan_for_training_config", "training_command",
]
