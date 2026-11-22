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
    validate_training_yaml_authoring,
)
from .config_document import TrainingConfigDocument
from .configs import (list_config_sources, open_existing_document, create_document_from_source)
from .datasets import (TrainingRecording, TrainingDatasetEntry, DatasetChangePreview,
                       list_training_datasets, training_recording_sources,
                       preview_dataset_change, apply_dataset_change)
from .training import (
    SavedTrainingPlan,
    plan_for_saved_training_config,
    plan_for_training_config,
    training_command,
    quick_hdn_training_command,
    quick_hdn_training_document,
)
from .workspace import create_workspace, validate_workspace
from .noise_models import NoiseModelEntry, list_noise_model_entries
from .jobs import TrainingLaunchRequest, JobSnapshot
from .inference import InferenceSelection, resolve_inference_selection
from .inference_io import (InferenceProvenance, TiffInputInfo, load_inference_tiff,
                           save_prediction_tiff)

__all__ = [
    "create_workspace", "validate_workspace",
    "find_new_completed_run", "get_installed_model_card", "get_installed_model_names",
    "list_hdn_noise_models", "list_reusable_training_datasets", "list_usable_training_runs",
    "list_user_training_configs", "managed_training_config_path", "save_training_yaml_text",
    "validate_training_yaml_text", "validate_training_yaml_authoring", "TrainingConfigDocument",
    "list_config_sources", "open_existing_document", "create_document_from_source",
    "TrainingRecording", "TrainingDatasetEntry", "DatasetChangePreview",
    "list_training_datasets", "training_recording_sources",
    "preview_dataset_change", "apply_dataset_change",
    "SavedTrainingPlan", "plan_for_saved_training_config",
    "quick_hdn_training_command", "quick_hdn_training_document", "plan_for_training_config",
    "training_command", "NoiseModelEntry", "list_noise_model_entries",
    "TrainingLaunchRequest", "JobSnapshot",
    "InferenceSelection", "resolve_inference_selection", "InferenceProvenance",
    "TiffInputInfo", "load_inference_tiff", "save_prediction_tiff",
]
