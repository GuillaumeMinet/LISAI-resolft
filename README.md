## LISAI: Light-efficient Imaging and Subsampled-image restoration with AI

*Also short for MoNaLISA + AI.*

This repository contains the code associated with our preprint:

> *RESOLFT time-lapse imaging empowered by deep learning*

This project implements a deep learning framework to enhance RESOLFT (Reversible Saturable Optical Fluorescence Transitions) time-lapse nanoscopy.  More specifically, we use the parallelized RESOLFT implementation **MoNaLISA** together with both denoising and up-sampling approaches to enable **prolonged imaging**, or a **4-fold speed improvement**, pushing the boundaries of **live-cell nanoscopy**.


# Package overview

LISAI is the Python package implementing the full image restoration workflow: dataset preprocessing,
model training, run tracking, evaluation, and applying trained models to image files. The current
codebase is centered on YAML-driven command line workflows and the package under `src/lisai`.

The main supported tasks are denoising and upsampling, with configuration examples for U-Net,
UNet-RCAN, RCAN, and LVAE/HDN-style models.

## Repository Layout

- `src/lisai/`: core package and source of truth for configuration, paths, preprocessing,
  training, evaluation, run tracking, and model loading
- `src/lisai_runner/`: optional local queue runner for training jobs
- `configs/`: project, data, preprocessing, training, inference, and schema YAML/JSON files
- `docs/`: architecture, quickstart, preprocessing, data organization, and run-selector notes
- `tests/`: unit tests for configuration, paths, training, evaluation, preprocessing, runs, and queue logic

## Environment

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

The CUDA environment currently pins `pytorch-cuda=12.1`. If your machine uses a different CUDA
setup, adjust the PyTorch packages accordingly.

## Local Configuration

LISAI resolves datasets, saved runs, inference outputs, and noise models from a local data root.
Create or edit `configs/local_config.yml` on your machine:

```yaml
infrastructure:
  data_root: "D:/path/to/lisai-data"
```

This file is local-machine configuration and is ignored by Git. If it is missing, importing the
package or running the CLI will prompt for a data root and write it.

With the default project config, the main data locations are:

- datasets: `<data_root>/datasets`
- noise models: `<data_root>/noise_models`
- training runs: `<data_root>/datasets/<dataset>/models/<models_subfolder>/<exp_name>`
- inference outputs: `<data_root>/datasets/<dataset>/inference/<inference_subfolder>/<exp_name>`

## Command Line Usage

Check that the CLI is installed:

```bash
lisai --help
lisai train --help
```

Run preprocessing from a tracked config:

```bash
lisai preprocess single
lisai preprocess configs/preprocess/single.yml
```

Run training from an example config:

```bash
lisai train examples/vim_denoising_unet
lisai train configs/training/examples/vim_denoising_unet.yml
```

Example training configs assume that the referenced preprocessed datasets already exist under your
configured data root.

Inspect and manage saved runs:

```bash
lisai runs list
lisai runs list --dataset vim_fixed
lisai runs plot <run-selector>
lisai runs open <run-selector>
```

Evaluate or apply a trained model:

```bash
lisai evaluate <run-selector> --split val --metrics psnr,ssim
lisai apply <run-selector> /path/to/images --tiling-size 512
lisai apply --run-id <run-id> /path/to/images
```

Continue an existing run in place:

```bash
lisai continue <run-selector>
lisai continue <run-selector> --yes
```

Run selectors can be a stable `--run-id`, a run directory name, a partial experiment name when it
is unambiguous, or `dataset[/models_subfolder]/run_dir_name`.

## Configuration

All main workflows are configured with YAML files:

- `configs/project_config.yml`: project-level path templates, run layout, naming, recovery, and queue defaults
- `configs/data_config.yml`: supported data formats, filename templates, and data subfolder names
- `configs/preprocess/*.yml`: preprocessing pipeline configs
- `configs/training/examples/*.yml`: tracked training examples
- `configs/training/local/*.yml`: local training configs, ignored by Git
- `configs/inference/*.yml`: defaults and named inference configs
- `configs/schema/*.json`: generated JSON schemas for supported config types

Short config names are resolved from their workflow folder. For example, `lisai train
examples/vim_denoising_unet` resolves under `configs/training/`, and `lisai preprocess single`
resolves under `configs/preprocess/`.

## Data Preparation

LISAI supports two training-data paths:

- Unprepared data: use this when the data has not been written by the LISAI preprocessing step.
  In the training config, set `data.prep_before: false`. The loader scans the configured dataset
  folder and creates a file-level split manifest during training. This path is currently intended
  for unpaired datasets.
- Preprocessed data: run a preprocessing config first, for example `lisai preprocess single`.
  The preprocess step writes the dataset into the expected training layout, can create
  `train`/`val`/`test` splits, and registers the result in the dataset registry under
  `<data_root>/datasets/dataset_registry.yml`. Training configs for this path use
  `data.prep_before: true` and `data.already_split: true`.

## Data And Runs

Filesystem paths are resolved through `src/lisai/infra/paths/paths.py`, not hardcoded throughout
the codebase. Preprocessing can update the dataset registry, training creates a run directory with
the resolved `config_train.yaml`, logs, checkpoints, validation images, and split metadata, and
evaluation reloads saved run metadata before building the inference runtime.

Supported data formats are currently:

- `single`
- `mltpl_snr`
- `timelapse`

Registered preprocessing pipelines include:

- `single_recon`
- `recon_mltpl_snr`
- `recon_timelapse_simple`

## Optional Queue Runner

The package also installs `lisai-runner`, a simple local queueing helper for training jobs. This
part of LISAI is still under active development, but it is useful for launching runs sequentially,
watching logs, and managing lightweight parameter sweeps.

```bash
lisai-runner queue submit --config examples/vim_denoising_unet
lisai-runner queue list
lisai-runner queue worker
```

For example, [`configs/training/examples/sweep_hdn_betaKL.yaml`](configs/training/examples/sweep_hdn_betaKL.yaml)
defines a sweep over `experiment.task.betaKL` values using a shared HDN base config:

```bash
lisai-runner queue submit-sweep --file examples/sweep_hdn_betaKL.yaml
```

The queue currently supports job submission, sweep submission, listing, logs, cancellation and cleanup.

## Documentation

- [Quick start](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Data organization](docs/data_organization.md)
- [Preprocessing](docs/preprocess.md)
- [Run selectors](docs/run_selectors.md)

## Development

Install the editable package, keep production changes focused under `src/lisai`, and run tests with:

```bash
pytest
```

For architecture-level changes, update the relevant docs and schemas alongside the code.
