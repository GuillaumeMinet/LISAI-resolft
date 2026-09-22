# LISAI: Light-efficient Imaging and Subsampled-image restoration with AI

*Also short for MoNaLISA + AI.*

This repository contains the code associated with our preprint:

> **RESOLFT time-lapse imaging empowered by deep learning**

This project implements a deep learning framework to enhance RESOLFT (Reversible Saturable Optical Fluorescence Transitions) time-lapse nanoscopy.  More specifically, we use the parallelized RESOLFT implementation **MoNaLISA** together with both denoising and up-sampling approaches to enable **prolonged imaging**, or a **4-fold speed improvement**, pushing the boundaries of **live-cell nanoscopy**.


## Package overview

LISAI is a Python framework for training and applying deep-learning image-restoration models to microscopy data. It was developed for the RESOLFT workflows described in our manuscript, but the training and inference infrastructure is designed to be reusable for other microscopy datasets.

Main supported workflows:

- denoising, including supervised and unsupervised HDN/LVAE training
- single-frame subsampled-image restoration
- multi-frame restoration using temporal context
- dataset preprocessing and registration
- experiment tracking and evaluation
- promotion, export, installation, and reuse of trained models

LISAI is driven primarily through YAML configuration files and the `lisai` command-line interface. 

While LISAI is structured as a Python package, the current release still assumes the repository layout and configuration files; support for fully standalone installation and use through a stable Python API is planned for a future release.


## Getting started

### Environment

Create one of the provided Conda environments from the repository root.

CPU:

```bash
conda env create -f environment.cpu.yml
conda activate lisai-cpu
```

CUDA:

```bash
conda env create -f environment.cuda.yml
conda activate lisai-cuda
```

Then install the package in editable mode:

```bash
pip install -e . --no-deps
```

The CUDA environment currently pins `pytorch-cuda=12.1`. If your machine uses a different CUDA setup, adjust the PyTorch packages accordingly.


### Local Configuration

LISAI resolves datasets, saved runs, inference outputs, and noise models from a local data root.

Create or edit `configs/local_config.yml` on your machine:

```yaml
infrastructure:
  data_root: "D:/path/to/lisai-data"
```

This file is local-machine configuration and is ignored by Git. If it is missing, importing the package or running the CLI will prompt for a data root and write it.

With the default project config, the main data locations are:

- datasets: `<data_root>/datasets` (split into `training/` and `evaluation/`)
- noise models: `<data_root>/noise_models`
- promoted models: `<data_root>/models`
- training runs: `<data_root>/datasets/training/<dataset>/runs/<models_subfolder>/<exp_name>`
- inference outputs: `<data_root>/inference/<source_name>/<model_name>`


## Typical workflow

### 1. Inspect or preprocess datasets

LISAI keeps a registry of datasets available under the configured data root.

```bash
lisai datasets list
lisai datasets list --training
lisai datasets list --evaluation
lisai datasets show <dataset>
```

> *Note*: LISAI differentiate between **training** and **evaluation-only** datasets. While a training dataset typically contains a test split which can be used to evaluate the trained models, an evaluation-only dataset contains only test data, typically acquired separately in specific conditions. 

Raw datasets can be converted into the standard LISAI training/evaluation layout with:

```bash
lisai preprocess single
lisai preprocess configs/preprocess/single.yml
```

Successful preprocessing registers the resulting dataset automatically.


### 2. Create a Training Configuration

Training presets provide starting points for the main supported tasks:

```bash
lisai configs list --kind preset
lisai configs new denoising_hdn_unsup --output local/my_hdn
```

Available presets include supervised and unsupervised HDN denoising, CARE and UNet-RCAN denoising, and single- or multi-frame upsampling.

Or, to start from the generic base template instead of a preset:

```bash
lisai configs new --custom --name my_experiment --output local/my_experiment
```

For registered dataset, you can include the dataset name at creation to pre-fill input/ground-truth fields:

```bash
lisai configs new denoising_hdn_unsup --output local/my_hdn --dataset <dataset-name>
```

You can then edit the configuration file: experiment name, saving parameters, training parameters, etc. You can validate the config and resolve it to see the final resolve config before launching training:

```
lisai configs validate local/my_hdn
lisai configs resolve local/my_hdn
```

> *Note*: To help edit configuration, YAML config files include JSON Schema definitions for autocompletion, type/value suggestions, inline validation, and documentation in compatible editors such as VS Code.


### 3. Train, inspect and manage training runs

Finally, you can run the training:

```
lisai train local/my_hdn
```

To inspect ongoing or finished runs:

```bash
lisai runs list
lisai runs list --dataset vim_fixed
lisai runs plot <run-selector>
lisai runs open <run-selector>
```

Runs store their resolved training configuration, checkpoints, logs, validation outputs, split information, and stable run ID.

To continue a run that stopped prematurely:

```bash
lisai continue <run-selector>
```

Run selectors can be a stable `--run-id`, a run directory name, a partial experiment name (if ambiguous, you will be prompted to choose between candidates runs) or `dataset[/models_subfolder]/run_dir_name`.


### 4. Evaluate or apply a trained model

By default, a run is evaluated on its test dataset as soon as the training finished.

To evaluate on another split, or a different evaluation-only dataset:

```bash
lisai evaluate <run-selector>               # test split by default
lisai evaluate <run-selector> --split val   # validation split
lisai evaluate <run-selector> --split train # training split
lisai evaluate <run-selector> --on <eval-dataset-name>
```

To **apply** a model:

```bash
lisai apply <run-selector> /path/to/images
```

>**NOTE**: The main difference between apply and evaluate is that evaluate expects data in the same form than the training data, while apply is to be used on "real application data". For example for the *upsampling task*, the test dataset contains fully-sampled data that is first downsampled, like in training, while apply expects real-world sub-sampled data. Also, evaluate will runs some metrics if a GT is available.

#### Inference configuration

`apply` and `evaluate` use typed, nested inference configurations, like training configurations.

Machine-local defaults live in:

```bash
configs/inference/local/defaults.yml
```

Additional sparse configurations can be stored under `configs/inference/`, including task-specific presets such as the HDN inference settings.

Configuration is resolved in the following order, with later values taking precedence:

```text
built-in defaults
    < local/defaults.yml
    < promoted-model inference config
    < selected inference config
    < explicit CLI options
```

The promoted-model layer applies when using `lisai apply --model ...`.

For example:

```bash
lisai apply --model my_hdn images/ --config my_inference
lisai apply <run-selector> images/ --tiling-size 512
```

#### Output handling

*evaluate* results are saved directly in the runs folder:
```bash
<run-name>/
└── evaluations/
   └── training_test/
      └── last_epoch/
      └── best_epoch/
   └── <evaluation-dataset-X>
```


*apply* outputs are by defaults saved to the normal LISAI inference directory. But for daily usage convenience, they can be saved closer to the the source data with 3 options:
- in_place: beside source_data, in same folder
- folder_inside: in a folder located inside the source_data root
- folder_outside: in a folder located outside the source_data root

You can specify your preference directly in the `local_config.yml` under the `inference` section, or for one-time usage, specify as a cli argument:

```bash
lisai apply <run-selector> images/ --output-mode in_place
```

Existing inference folders can also be resumed safely:

```bash
lisai apply <run-selector> images/ --skip-existing
lisai apply <run-selector> images/ --reuse-folder
```

Check typed inference configs, or use `apply -h` and `evaluate -h` for full options.

### 5. Promote a trained run to a reusable model

Completed runs can be detached from the experiment hierarchy and promoted into the local model library:

```bash
lisai runs promote <run-selector> --name my_model
lisai models list
lisai models show my_model
```

Promoted models can be used directly for inference:

```bash
lisai apply --model my_model /path/to/images
```

A model-specific inference configuration can be attached during promotion or afterwards:

```bash
lisai runs promote <run-selector> \
    --name my_hdn \
    --inference-config presets/hdn

lisai models set-config my_hdn presets/hdn
```

Models can also be packaged for transfer:

```bash
lisai models export my_hdn
lisai models install /path/to/my_hdn.lisai.zip
```

LISAI also supports a downloadable model catalog:

```bash
lisai models catalog
lisai models download <model-name> --install
```


## Concepts and configuration reference

### Configuration

All main workflows are configured with YAML files:

- `configs/project_config.yml`: project-level path templates, run layout, naming, recovery, and queue defaults
- `configs/data_config.yml`: supported data formats, filename templates, and data subfolder names
- `configs/preprocess/*.yml`: preprocessing pipeline configs
- `configs/training/presets/*.yml`: tracked recipes used to create local configs
- `configs/training/templates/*.yml`: tracked generic starting points with placeholders
- `configs/training/examples/*.yml`: tracked training examples
- `configs/training/local/*.yml`: local training configs, ignored by Git
- `configs/inference/local/*.yml`: personal inference defaults and workflow overrides, ignored by Git
- `configs/inference/**/*.yml`: tracked inference configs may stay sparse; unspecified values inherit from `local/defaults.yml`
- `configs/schema/*.json`: generated JSON schemas for supported config types

Short config names are resolved from their workflow folder. For example, `lisai train examples/vim_denoising_unet` resolves under `configs/training/`, and `lisai preprocess single` resolves under `configs/preprocess/`. Inference config names check `configs/inference/local/` first, then `configs/inference/`. Explicit values in the selected inference config override `local/defaults.yml`, and explicit CLI options override both.

Training presets and templates are catalog inputs, not direct training inputs. Use `lisai configs new ...` to instantiate a preset or template into `configs/training/local/`, then train the local config.


### Data Preparation

LISAI supports two training-data paths:

- Unprepared data: use this when the data has not been written by the LISAI preprocessing step. In the training config, set `data.prep_before: false`. The loader scans the configured dataset folder and creates a file-level split manifest during training. This path is currently intended for unpaired datasets.

- Preprocessed data: run a preprocessing config first, for example `lisai preprocess single`. The preprocess step writes the dataset into the expected training layout, can create `train`/`val`/`test` splits, and registers the result in the dataset registry under `<data_root>/datasets/dataset_registry.yml`. Training configs for this path use `data.prep_before: true` and `data.already_split: true`.

See [Preprocessing](docs/preprocess.md) and [Data organization](docs/data_organization.md) for the detailed folder semantics.


### Data And Runs

Filesystem paths are resolved through `src/lisai/infra/paths/paths.py`, not hardcoded throughout the codebase. Preprocessing can update the dataset registry, training creates a run directory with the resolved `config_train.yaml`, logs, checkpoints, validation images, and split metadata, and evaluation reloads saved run metadata before building the inference runtime.

Supported data formats are currently:

- `single`
- `mltpl_snr`
- `timelapse`

Registered preprocessing pipelines include:

- `single_recon`
- `recon_mltpl_snr`
- `recon_timelapse_simple`


## Additional tools

### Optional Queue Runner

The package also installs `lisai-runner`, a simple local queueing helper for training jobs. This part of LISAI is still under active development, but it is useful for launching runs sequentially, watching logs, and managing lightweight parameter sweeps.

```bash
lisai-runner queue submit --config examples/vim_denoising_unet
lisai-runner queue list
lisai-runner queue worker
```

For example, [`configs/training/examples/sweep_hdn_betaKL.yaml`](configs/training/examples/sweep_hdn_betaKL.yaml) defines a sweep over `experiment.task.betaKL` values using a shared HDN base config:

```bash
lisai-runner queue submit-sweep --file examples/sweep_hdn_betaKL.yaml
```

The queue currently supports job submission, sweep submission, listing, logs, cancellation and cleanup.


### Reproducibility and external baselines

The repository also contains utilities for importing predictions produced by external implementations into the LISAI evaluation layout. This is used, for example, to compare external baselines with LISAI runs using the same evaluation and plotting code.

Imported external runs preserve their checkpoint/configuration provenance and evaluation outputs, but they are not executable LISAI training runs: commands such as `continue`, `apply`, and `evaluate` do not operate on them.

These utilities currently live under `src/scripts/` and are primarily intended for reproducing the analyses associated with this work.


## Documentation

- [Quick start](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Data organization](docs/data_organization.md)
- [Preprocessing](docs/preprocess.md)
- [Run selectors](docs/run_selectors.md)


## Repository Layout

- `src/lisai/`: core package and source of truth for configuration, paths, preprocessing, training, evaluation, run tracking, and model loading
- `src/lisai_runner/`: optional local queue runner for training jobs
- `src/scripts/`: reusable utility scripts such as schema generation, legacy model import, etc.
- `configs/`: project, data, preprocessing, training, inference, and schema YAML/JSON files
- `docs/`: architecture, quickstart, preprocessing, data organization, and run-selector notes
- `tests/`: unit tests for configuration, paths, training, evaluation, preprocessing, runs, and queue logic
- `graphs/`: scripts to generate graphs from the publication.


## Development

Install the editable package, keep production changes focused under `src/lisai`, and run tests with:

```bash
pytest
```

For architecture-level changes, update the relevant docs and schemas alongside the code.
