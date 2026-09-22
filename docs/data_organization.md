# Data Organization

Filesystem layout in LISAI is defined through [`Paths`](../src/lisai/infra/paths/paths.py), not through hardcoded strings.

## Main Locations

- Dataset registry: `Paths.dataset_registry_path()`
- Local data root: `Paths.data_root()`
- Training datasets root: `Paths.training_datasets_root()`
- Evaluation-only datasets root: `Paths.evaluation_datasets_root()`
- Dataset root for loading: `Paths.dataset_dir(dataset_name=..., data_subfolder=..., usage=...)`
- Training run folder: `Paths.run_dir(dataset_name=..., models_subfolder=..., exp_name=...)`
- Imported external run folder: `Paths.external_run_dir(dataset_name=..., run_name=...)`
- Promoted-model registry: `Paths.promoted_model_registry_path()`
- Promoted-model folder: `Paths.promoted_model_dir(model_name=...)`
- General inference root: `Paths.inference_root()`
- Default `apply` output folder: `Paths.inference_output_dir(source_name=..., model_name=...)`
- Noise model file: `Paths.noise_model_path(...)`
- TensorBoard runs: `Paths.tensorboard_dir(...)`

With the default project configuration, the data root is organized as:

```text
<data_root>/
├── datasets/
│   ├── dataset_registry.yml
│   ├── training/
│   │   └── <dataset>/
│   │       ├── dump/
│   │       ├── preprocess/
│   │       ├── runs/
│   │       │   └── <models_subfolder>/
│   │       │       └── <exp_name>/
│   │       └── external_runs/
│   │           └── <external_run>/
│   └── evaluation/
│       └── <dataset>/
│           ├── dump/
│           └── preprocess/
├── models/
│   ├── model_registry.yml
│   └── <promoted_model>/
├── inference/
│   └── <source_name>/
│       └── <model_name>/
└── noise_models/
    └── <noise_model>/
```

The physical `training` and `evaluation` dataset folder names, the run-container name (`runs` by default), and the main root templates come from `configs/project_config.yml`.

## Training Run Layout

Inside a run folder, `Paths` resolves the standard training artifacts:

- `config_train.yaml`: `Paths.cfg_train_path(...)`
- training log: `Paths.log_file_path(...)`
- loss file: `Paths.loss_file_path(...)`
- loss plot: `Paths.loss_plot_path(...)`
- checkpoints folder: `Paths.checkpoints_dir(...)`
- validation images folder: `Paths.validation_images_dir(...)`
- split manifest: `Paths.split_manifest_path(...)`
- retrain origin folder: `Paths.retrain_origin_dir(...)`

Dataset-based evaluation results are kept with the source run rather than in the general inference root:

```text
<run_dir>/evaluations/
├── training_<split>/
│   └── <checkpoint_selection>/
└── <evaluation_dataset>/
    └── <checkpoint_selection>/
```

`lisai apply`, by contrast, uses the general inference root by default (`<data_root>/inference/...`), unless its output mode is overridden by the local/inference config or CLI.

## Preprocess Layout

Preprocess paths are also resolved through `Paths` and are usage-aware:

- raw dump area: `Paths.dataset_dump_dir(..., usage=...)`
- processed dataset root: `Paths.dataset_preprocess_dir(..., usage=...)`
- individual processed files: `Paths.preprocessed_image_full_path(..., usage=...)`

Training datasets may contain `train`/`val`/`test` folders under their processed outputs. Evaluation-only datasets are stored separately under `datasets/evaluation/` and are treated as whole-dataset evaluation resources.

## Promoted Models

Reusable promoted models live outside the training-run hierarchy under `<data_root>/models/`. The registry is stored at `model_registry.yml`, while each model has its own directory containing its LISAI model manifest, weights, model card, and any packaged inference/noise-model artifacts required by that model.

## External Runs

Imported external baselines are stored under the training dataset they were trained on:

```text
<data_root>/datasets/training/<dataset>/external_runs/<run_name>/
```

They preserve external checkpoint/config provenance and imported evaluation outputs, but they are not executable LISAI training runs.

## Practical Rule

If code needs a filesystem location, prefer adding or using a `Paths` method instead of rebuilding the path manually. Evaluation result subfolders that are intentionally run-local are created by the evaluation layer from the selected run directory.
