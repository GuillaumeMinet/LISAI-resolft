# Data Organization

Filesystem layout in LISAI is defined through [`Paths`](../src/lisai/infra/paths/paths.py), not through hardcoded strings.

## Main Locations

- Dataset registry: `Paths.dataset_registry_path()`
- Dataset root for loading: `Paths.dataset_dir(dataset_name=..., data_subfolder=...)`
- Training run folder: `Paths.run_dir(dataset_name=..., models_subfolder=..., exp_name=...)`
- TensorBoard runs: `Paths.tensorboard_dir(...)`
- Evaluation outputs: `Paths.inference_dir(...)`
- Noise model file: `Paths.noise_model_path(...)`

With the default project configuration, training runs live under:
`datasets/<dataset>/models/<models_subfolder>/<exp_name>`

## Training Run Layout

Inside a run folder, `Paths` also resolves the standard artifacts:

- `config_train.yaml`: `Paths.cfg_train_path(...)`
- training log: `Paths.log_file_path(...)`
- loss file: `Paths.loss_file_path(...)`
- loss plot: `Paths.loss_plot_path(...)`
- checkpoints folder: `Paths.checkpoints_dir(...)`
- validation images folder: `Paths.validation_images_dir(...)`
- split manifest: `Paths.split_manifest_path(...)`
- retrain origin folder: `Paths.retrain_origin_dir(...)`

## Preprocess Layout

Preprocess outputs are also resolved through `Paths`:

- raw dump area: `Paths.dataset_dump_dir(...)`
- processed dataset root: `Paths.dataset_preprocess_dir(...)`
- individual processed files: `Paths.preprocessed_image_full_path(...)`

## Practical Rule

If code needs a filesystem location, prefer adding or using a `Paths` method instead of rebuilding the path manually.
