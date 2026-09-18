from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint_naming import model_filename


@dataclass(frozen=True)
class TemplateKeys:
    dataset_registry: str = "dataset_registry"
    dataset_loading: str = "dataset_loading"
    run_dir: str = "run_dir"
    tensorboard_runs_dir: str = "tensorboard_runs_dir"
    inference_output_dir: str = "inference_output_dir"
    noise_model: str = "noise_model"
    promoted_model_registry: str = "promoted_model_registry"
    promoted_model_dir: str = "promoted_model_dir"
    promoted_model_exports_dir: str = "promoted_model_exports_dir"
    promoted_model_downloads_dir: str = "promoted_model_downloads_dir"


class Paths:
    """ 
    Paths provides the canonical API to resolve filesystem locations.

    It formats path templates defined in project configuration using
    runtime context (dataset, run, etc.) and centralizes all filesystem
    structure logic to avoid hardcoded paths across the codebase.
    """
    def __init__(self, stg=None, keys: TemplateKeys | None = None):
        if stg is None:
            from lisai.config.settings import settings as default_settings

            stg = default_settings
        self.settings = stg
        self.keys = keys or TemplateKeys()

    # Canonical template paths
    def project_root(self) -> Path:
        return Path(self.settings.PROJECT_ROOT).resolve()

    def datasets_root(self) -> Path:
        return Path(self.settings.resolve_path(self.settings.project_cfg.paths.roots["data_dir"])).resolve()

    def dataset_usage_subfolder(self, usage: str) -> str:
        """Return the configured physical folder name for a semantic dataset usage."""
        usage_key = str(usage).strip().lower()
        mapping = self.settings.project_cfg.paths.dataset_usage_subfolders
        try:
            return mapping[usage_key]
        except KeyError as exc:
            known = ", ".join(sorted(mapping))
            raise ValueError(
                f"Unknown dataset usage {usage!r}. Configured usages: {known}."
            ) from exc

    def dataset_usage_root(self, usage: str) -> Path:
        return self.datasets_root() / self.dataset_usage_subfolder(usage)

    def training_datasets_root(self) -> Path:
        return self.dataset_usage_root("training")

    def evaluation_datasets_root(self) -> Path:
        return self.dataset_usage_root("evaluation")

    def run_container_dirname(self) -> str:
        """ Returns the folder name used to saved runs (e.g. "runs")"""
        roots = self.settings.project_cfg.paths.roots or {}
        raw = roots.get("run_container_dirname", "runs")
        text = str(raw).strip().strip("/\\")
        if not text:
            raise ValueError("project.paths.roots.run_container_dirname must not be empty.")
        if "/" in text or "\\" in text:
            raise ValueError("project.paths.roots.run_container_dirname must be a single directory name.")
        return text

    def dataset_registry_path(self) -> Path:
        return self.settings.get_template_path(self.keys.dataset_registry)

    def promoted_models_root(self) -> Path:
        roots = self.settings.project_cfg.paths.roots or {}
        template = roots.get("promoted_models_dir", "{data_root}/models")
        return Path(self.settings.resolve_path(template)).resolve()

    def promoted_model_registry_path(self) -> Path:
        return self.settings.get_template_path(self.keys.promoted_model_registry)

    def promoted_model_dir(self, *, model_name: str) -> Path:
        return self.settings.get_template_path(
            self.keys.promoted_model_dir,
            model_name=model_name,
        )

    def promoted_model_exports_dir(self) -> Path:
        return self.settings.get_template_path(self.keys.promoted_model_exports_dir)

    def promoted_model_downloads_dir(self) -> Path:
        return self.settings.get_template_path(self.keys.promoted_model_downloads_dir)

    def dataset_dir(
        self,
        *,
        dataset_name: str,
        data_subfolder: str = "",
        usage: str = "training",
    ) -> Path:
        return self.settings.get_template_path(
            self.keys.dataset_loading,
            dataset_name=dataset_name,
            data_subfolder=data_subfolder,
            dataset_usage_subfolder=self.dataset_usage_subfolder(usage),
        )

    def dataset_readme_path(
        self,
        *,
        dataset_name: str,
        data_subfolder: str = "",
        usage: str = "training",
    ) -> Path:
        """ Returns full dataset readme path"""
        dataset_path = self.dataset_dir(
            dataset_name=dataset_name,
            data_subfolder=data_subfolder,
            usage=usage
        )
        filename = "readme.txt"
        return dataset_path / filename

    def dataset_runs_dir(self, *, dataset_name: str) -> Path:
        """Return saved runs directory for a training dataset."""
        return self.dataset_dir(dataset_name=dataset_name, usage="training") / self.run_container_dirname()
    
    def dataset_runs_dir_from_dataset_dir(self, dataset_dir: str | Path) -> Path:
        """ Returns saved runs directory for a given dataset path."""
        return Path(dataset_dir) / self.run_container_dirname()

    def external_run_container_dirname(self) -> str:
        """Return the folder name used for imported external runs."""
        return "external_runs"

    def dataset_external_runs_dir(self, *, dataset_name: str) -> Path:
        """Return imported external runs directory for a training dataset."""
        return self.dataset_dir(dataset_name=dataset_name, usage="training") / self.external_run_container_dirname()

    def dataset_external_runs_dir_from_dataset_dir(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / self.external_run_container_dirname()

    def external_run_dir(self, *, dataset_name: str, run_name: str) -> Path:
        return self.dataset_external_runs_dir(dataset_name=dataset_name) / run_name

    def run_dir(self, *, dataset_name: str, models_subfolder: str, exp_name: str) -> Path:
        """Return a training run directory for a dataset, subfolder and experiment name."""
        return self.settings.get_template_path(
            self.keys.run_dir,
            dataset_name=dataset_name,
            dataset_usage_subfolder=self.dataset_usage_subfolder("training"),
            run_container_dirname=self.run_container_dirname(),
            models_subfolder=models_subfolder,
            exp_name=exp_name,
        )

    def tensorboard_dir(self, *, dataset_name: str, tensorboard_subfolder: str = "") -> Path:
        return self.settings.get_template_path(
            self.keys.tensorboard_runs_dir,
            dataset_name=dataset_name,
            tensorboard_subfolder=tensorboard_subfolder,
        )

    def inference_root(self) -> Path:
        """Return the effective inference root after local overrides are applied."""
        return Path(self.settings.resolve_path("{paths.roots.inference_dir}")).resolve()

    def inference_output_dir(self, *, source_name: str, model_name: str) -> Path:
        """Return the default output directory for one apply invocation."""
        return self.settings.get_template_path(
            self.keys.inference_output_dir,
            source_name=source_name,
            model_name=model_name,
        )

    def noise_model_path(self, *, noiseModel_name: str) -> Path:
        return self.settings.get_template_path(
            self.keys.noise_model,
            noiseModel_name=noiseModel_name,
        )

    def noise_model_dir(self, *, noiseModel_name: str) -> Path:
        return self.noise_model_path(noiseModel_name=noiseModel_name).parent

    def noise_model_norm_prm_path(self, *, noiseModel_name: str) -> Path:
        return self.noise_model_dir(noiseModel_name=noiseModel_name) / "norm_prm.json"

    # Run layout subdirectories
    def _subdir(self, run_dir: str | Path, key: str, default: str) -> Path:
        subdirs = self.settings.project_cfg.run_layout.subdirs
        name = subdirs.get(key, default)
        return Path(run_dir) / name

    def checkpoints_dir(self, *, run_dir: str | Path) -> Path:
        return self._subdir(run_dir, "checkpoints", "checkpoints")

    def validation_images_dir(self, *, run_dir: str | Path) -> Path:
        return self._subdir(run_dir, "validation_images", "validation_images")

    def retrain_origin_dir(self, *, run_dir: str | Path) -> Path:
        return self._subdir(run_dir, "retrain_origin", "retrain_origin")

    # Run layout artifact files
    def _artifact(self, run_dir: str | Path, key: str, default: str) -> Path:
        artifacts = self.settings.project_cfg.run_layout.artifacts
        name = artifacts.get(key, default)
        return Path(run_dir) / name

    def loss_file_path(self, *, run_dir: str | Path) -> Path:
        return self._artifact(run_dir, "loss_file", "loss.txt")

    def log_file_path(self, *, run_dir: str | Path) -> Path:
        return self._artifact(run_dir, "train_log", "train_log.log")

    def cfg_train_path(self, *, run_dir: str | Path) -> Path:
        return self._artifact(run_dir, "config_train", "config_train.yaml")

    def split_manifest_path(self, *, run_dir: str | Path) -> Path:
        return self._artifact(run_dir, "split_manifest", "split_manifest.json")

    def loss_plot_path(self, *, run_dir: str | Path) -> Path:
        return self._artifact(run_dir, "loss_plot", "loss_plot.png")

    # retrain origin artifacts
    def _origin_artifact(self, run_dir: str | Path, key: str, default: str) -> Path:
        artifacts = self.settings.project_cfg.run_layout.retrain_origin_artifacts
        name = artifacts.get(key, default)
        return self.retrain_origin_dir(run_dir=run_dir) / name

    def retrain_origin_loss_path(self, *, run_dir: str | Path) -> Path:
        return self._origin_artifact(run_dir, "loss_file", "origin_loss.txt")

    def retrain_origin_log_path(self, *, run_dir: str | Path) -> Path:
        return self._origin_artifact(run_dir, "train_log", "origin_log.log")

    def retrain_origin_cfg_path(self, *, run_dir: str | Path) -> Path:
        return self._origin_artifact(run_dir, "config_train", "origin_config.yaml")

    # preprocess paths
    def dataset_dump_dir(
        self,
        *,
        dataset_name: str,
        data_type: str = "",
        additional_subfolder: str = "",
        usage: str = "training",
    ):
        """dataset_dir / dump"""
        ds_path = self.dataset_dir(dataset_name=dataset_name, usage=usage)
        dump_subfolder = self.settings.data_cfg.subfolders.get("dump","dump")
        return ds_path / dump_subfolder / data_type / additional_subfolder

    def dataset_preprocess_dir(
        self, *, dataset_name: str, data_type: str = "", usage: str = "training"
    ):
        """dataset_dir / preprocess"""
        ds_path = self.dataset_dir(dataset_name=dataset_name, usage=usage)
        preprocess_subfolder = self.settings.data_cfg.subfolders.get("preprocess","preprocess")
        return ds_path / preprocess_subfolder / data_type

    def preprocess_log_path(
        self, *, dataset_name: str, data_type: str, usage: str = "training"
    ) -> Path:
        key = f"{data_type}_preprocess"
        filename = self.settings.data_cfg.logs.get(key)
        if filename is None:
            raise KeyError(f"Unknown preprocess log key '{key}' in data config.")
        return self.dataset_preprocess_dir(
            dataset_name=dataset_name, data_type=data_type, usage=usage
        ) / filename
    
    def preprocessed_image_full_path(
        self,
        *,
        dataset_name: str,
        fmt: str,
        data_type: str = "",
        additional_subfolder: str = "",
        usage: str = "training",
        **kwargs,
    ):
        """ 
        dataset_dir / preprocess / subfolder / filename
        e.g. : <data_root>/preprocess/gt_avg/c05.tif
        """
        base_dir = self.dataset_preprocess_dir(
            dataset_name=dataset_name, data_type=data_type, usage=usage
        )
        filename =  self.settings.get_data_filename(fmt=fmt,data_type=data_type,**kwargs)
        return base_dir / additional_subfolder / filename
    
    # Checkpoint path helper
    def checkpoint_path(
        self,
        *,
        run_dir: str | Path,
        load_method: str | None = None,
        best_or_last: str | None = None,
        train_mode: str | None = None,
        epoch_number: int | None = None,
        model_name: str | None = None,
    ) -> Path:
        run_dir = Path(run_dir)
        if not model_name:
            if not load_method:
                raise ValueError("checkpoint_path requires load_method when model_name is not provided.")
            model_name = model_filename(
                load_method=load_method,
                best_or_last=best_or_last,
                train_mode=train_mode,
                epoch_number=epoch_number,
            )
        return self.checkpoints_dir(run_dir=run_dir) / model_name
