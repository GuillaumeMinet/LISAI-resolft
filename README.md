# LISAI: Light-efficient Imaging and Subsampled-image restoration with AI

*Also short for MoNaLISA + AI.*

This repository contains the code associated with our preprint:

> **Time-lapse RESOLFT nanoscopy empowered by deep learning**

This project implements a deep learning framework to enhance RESOLFT (Reversible Saturable Optical Fluorescence Transitions) time-lapse nanoscopy.  More specifically, we use the parallelized RESOLFT implementation **MoNaLISA** together with both denoising and up-sampling approaches to enable **prolonged imaging**, or a **4-fold speed improvement**, pushing the boundaries of **live-cell nanoscopy**.

> Example datasets and trained models used in the manuscript, are available from the [accompanying Zenodo repository](https://doi.org/10.5281/zenodo.22877255), and can be directly downloaded via lisai command line (see **Download paper datasets and models** section below).

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

LISAI v0.1 supports explicit standalone workspaces and source-checkout developer configurations. Built-in presets are packaged with the installed distribution. The file-based Python API remains available; streamlined inference APIs and minimal-install dependency validation are planned for M3.


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

LISAI resolves datasets, saved runs, inference outputs and noise models from an
explicit workspace. Initialize it once, without modifying the installed package:

```bash
lisai workspace init D:/path/to/lisai-workspace
export LISAI_HOME=D:/path/to/lisai-workspace
lisai settings show
```

Its authoritative settings live in `<workspace>/configs/workspace.yml` and
`<workspace>/configs/data.yml`. For editable development, use `lisai dev init`
so experiments remain under the source checkout's `configs/`. Use `lisai settings set` to persist developer selection; ordinary commands have no live `--workspace` or `--config-root` overrides.
Neither imports nor ordinary CLI commands prompt or write configuration.
See [workspace setup and migration](docs/workspaces.md).

With the default workspace config, the main data locations are:

- datasets: `<workspace_root>/datasets` (split into `training/` and `evaluation/`)
- noise models: `<workspace_root>/noise_models`
- promoted models: `<workspace_root>/models`
- training runs: `<workspace_root>/datasets/training/<dataset>/runs/<models_subfolder>/<exp_name>`
- inference outputs: `<workspace_root>/inference/<source_name>/<model_name>`


### Download paper datasets and models

To start with concrete examples, you can directly use datasets available from the paper Zenodo repository.
LISAI can browse, download and install the published datasets directly into the selected workspace:

```bash
lisai datasets catalog
lisai datasets download <dataset-name>
lisai datasets download --all
```
If dataset archives have already been downloaded manually from Zenodo, they can
instead be installed from a file or folder:

```bash
lisai datasets install <path-to-archive-or-folder>
```

Published promoted models can similarly be downloaded and installed with:

```bash
lisai models catalog
lisai models download <model-name> --install
lisai models download --all --install
```

## Typical workflow

### 1. Inspect or preprocess datasets

LISAI keeps a registry of datasets inside the selected workspace.

```bash
lisai datasets list
lisai datasets list --training
lisai datasets list --evaluation
lisai datasets show <dataset>
```

> *Note*: LISAI differentiate between **training** and **evaluation-only** datasets. While a training dataset typically contains a test split which can be used to evaluate the trained models, an evaluation-only dataset contains only test data, typically acquired separately in specific conditions. 

Raw datasets can be converted into the standard LISAI training/evaluation layout with:

```bash
lisai configs list --domain preprocess
lisai configs new preprocess single_recon --dataset <dataset-name>
# Edit the generated <config-root>/preprocess/<dataset-name>.yml
lisai preprocess <dataset-name>
```

Successful preprocessing registers the resulting dataset automatically.

### 2. Create a Training Configuration

Training presets provide starting points for the main supported tasks:

```bash
lisai configs list --kind preset
lisai configs new denoising_hdn_unsup --output my_hdn
```

Available presets include supervised and unsupervised HDN denoising, CARE and UNet-RCAN denoising, and single- or multi-frame upsampling.

Or, to start from the generic base template instead of a preset:

```bash
lisai configs new --custom --name my_experiment --output my_experiment
```

For registered dataset, you can include the dataset name at creation to pre-fill input/ground-truth fields:

```bash
lisai configs new denoising_hdn_unsup --output my_hdn --dataset <dataset-name>
```

You can then edit the configuration file: experiment name, saving parameters, training parameters, etc. You can validate the config and resolve it to see the final resolve config before launching training:

```
lisai configs validate my_hdn
lisai configs resolve my_hdn
```

> *Note*: To help edit configuration, YAML config files include JSON Schema definitions for autocompletion, type/value suggestions, inline validation, and documentation in compatible editors such as VS Code.


### 3. Train, inspect and manage training runs

Finally, you can run the training:

```
lisai train my_hdn
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
configs/inference/defaults.yml
```

Additional sparse configurations can be stored under `configs/inference/`, including task-specific presets such as the HDN inference settings.

Configuration is resolved in the following order, with later values taking precedence:

```text
built-in defaults
    < inference/defaults.yml
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

Set workspace-wide inference preferences in `<workspace>/configs/workspace.yml`, or use whitelisted per-developer overrides in `<config-root>/local.yml`. For a single invocation, specify the CLI argument:

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

For now, models catalog only includes models linked to the original publication and its Zenodo repository. Future version might include a shared workflow to upload models, if needs be.


## Concepts and configuration reference

### Configuration

All main workflows are configured with YAML files:

- `<workspace>/configs/workspace.yml`: authoritative paths, run layout, naming, recovery and queue defaults
- `<workspace>/configs/data.yml`: supported data formats and filename templates
- `lisai` package resources: read-only training presets/templates/examples, inference presets and generic preprocess templates
- `<config-root>/training/*.yml`: editable training experiments
- `<config-root>/preprocess/*.yml`: editable preprocessing configurations
- `<config-root>/inference/*.yml`: editable inference defaults and post-training settings
- `<config-root>/schema/*.json`: explicitly refreshed, registry-aware editor schemas

The writable config root defaults to `<workspace>/configs`, but can be the
verified editable checkout's `configs/` in developer mode.

Short config names are resolved from their workflow folder. For example, `lisai train examples/vim_denoising_unet` selects a packaged example, while `lisai train my_training` selects an editable YAML from the selected writable root. Preprocess templates must first be instantiated with `lisai configs new preprocess <template> --dataset <name>`. Inference user files are editable, and `presets/` refer to packaged inference resources. Explicit values in the selected inference config override `defaults.yml`, and explicit CLI options override both.

Training presets and templates are catalog inputs, not direct training inputs. Use `lisai configs new ...` to instantiate a preset or template into `configs/training/`, then train the local config.


### Data Preparation

LISAI supports two training-data paths:

- Unprepared data: use this when the data has not been written by the LISAI preprocessing step. In the training config, set `data.prep_before: false`. The loader scans the configured dataset folder and creates a file-level split manifest during training. This path is currently intended for unpaired datasets.

- Preprocessed data: run a local preprocessing configuration first, for example `lisai preprocess my_preprocess`. The preprocess step writes the dataset into the expected training layout, can create `train`/`val`/`test` splits, and registers the result in the dataset registry under `<workspace_root>/datasets/dataset_registry.yml`. Training configs for this path use `data.prep_before: true` and `data.already_split: true`.

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

For example, the packaged `examples/sweep_hdn_betaKL` defines a sweep over `experiment.task.betaKL` values using a shared HDN base config:

```bash
lisai-runner queue submit-sweep --file examples/sweep_hdn_betaKL.yaml
```

The queue currently supports job submission, sweep submission, listing, logs, cancellation and cleanup. By default, queue state lives at `<workspace>/.lisai/queue/` (override with `LISAI_QUEUE_ROOT`), and TensorBoard output is under `<workspace>/tensorboard/`. The worker records both the submitted workspace and writable config root, so a subprocess does not depend on the source checkout or its working directory.


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
