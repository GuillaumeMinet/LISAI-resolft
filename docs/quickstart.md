# LISAI Quick Start

This guide covers the shortest path from clone to a first training or evaluation run.

## Environment

Create one Conda environment from the repo root:

```powershell
conda env create -f environment.cpu.yml
conda activate lisai-cpu
```

or:

```powershell
conda env create -f environment.cuda.yml
conda activate lisai-cuda
```

Install the package in editable mode:

```powershell
pip install -e . --no-deps
```

## Local Data Root

LISAI resolves datasets, saved runs, promoted models, inference outputs, and noise models from a local data root. Create `configs/local_config.yml` on your machine:

```yaml
infrastructure:
  data_root: "D:/path/to/lisai-data"
```

This file is intentionally ignored by Git. If it is missing, the first CLI command prompts for a data root and writes the file.

## Smoke Test

Check that the CLI is available:

```powershell
lisai --help
lisai train --help
```

## Datasets

Inspect datasets already registered under the local data root:

```powershell
lisai datasets list
lisai datasets list --training
lisai datasets list --evaluation
lisai datasets show <dataset>
```

Training datasets and evaluation-only datasets are stored separately. A training dataset can provide train/validation/test splits, while an evaluation-only dataset is consumed as one complete independent evaluation resource.

## Training

List available training configs:

```powershell
lisai configs list
lisai configs list --kind preset
```

Create a local editable config from a preset:

```powershell
lisai configs new denoising_hdn_unsup --name my_hdn
# or to specify the configs subfolder:
lisai configs new denoising_hdn_unsup --output local/my_hdn

# Registered datasets can also pre-fill dataset-dependent fields:
lisai configs new denoising_hdn_unsup --output local/my_hdn --dataset <dataset>

# Edit configs/training/local/my_hdn.yml and replace CHANGEME values.
lisai configs validate local/my_hdn
lisai configs resolve local/my_hdn
```

To start from the generic base template instead:

```powershell
lisai configs new --custom --name my_experiment --output local/my_experiment
```

If `configs new` warns about `CHANGEME`, those fields must be edited before `configs validate` or `train` can succeed.

Run a local config or a public example config:

```powershell
lisai train local/my_hdn
lisai train examples/vim_denoising_unet
```

You can also pass an explicit file path:

```powershell
lisai train configs/training/examples/vim_denoising_unet.yml
```

Training resolves the config, creates a run directory, saves `config_train.yaml`, and writes checkpoints and logs under the run folder. Presets and templates must be instantiated into `configs/training/local/` before training. Example configs assume the referenced datasets already exist under your configured data root.

## Evaluation and Apply

Use the CLI for run-based evaluation and inference:

```powershell
lisai evaluate Gag/Upsamp/my_model_00 --split val --metrics psnr,ssim
lisai evaluate Gag/Upsamp/my_model_00 --on gag_independent --metrics psnr,ssim
lisai apply Gag/Upsamp/my_model_00 /data/images --tiling-size 512
lisai apply --run-id 01ARZ3NDEKTSV4RRFFQ69G7ACD /data/images
```

`evaluate` rebuilds dataset inputs from the selected run and either one of its train/val/test splits or a registered evaluation-only dataset. Its results are stored under the source run's `evaluations/` directory. `apply` consumes normal input files/directories and uses the configured inference output routing.

Both commands accept `--config <name>` for settings under `configs/inference/`. Inference settings are resolved in this order, with later layers taking precedence:

```text
built-in typed defaults
    < configs/inference/local/defaults.yml
    < promoted-model inference config (apply --model only)
    < selected inference config
    < explicit CLI options
```

`tiling_size: auto` uses the saved model default, a positive integer forces a tile size, and `tiling_size: off` or `--no-tiling` disables tiling.

Run selectors must refer to a discovered LISAI training run, either by `--run-id`, `dataset[/subfolder]/run_dir_name`, `run_dir_name`, or a partial experiment name. See [`docs/run_selectors.md`](run_selectors.md) for details.

## Promoted Models

A completed or stopped training run can be promoted into the reusable local model library:

```powershell
lisai runs promote my_model_00 --name my_model
lisai models list
lisai models show my_model
```

Promoted models can be applied without referring back to the training-run hierarchy:

```powershell
lisai apply --model my_model /data/images
```

They can also carry a model-specific inference config and can be exported, installed, or downloaded from the LISAI model catalog. Use `lisai models --help` and the README for the full model-management workflow.

## Preprocess

Run preprocessing from a YAML config:

```powershell
lisai preprocess single
lisai preprocess configs/preprocess/single.yml
```

Tracked preprocess examples are available in `configs/preprocess/`. They assume the corresponding dataset dump folders already exist under your configured data root. Successful preprocessing updates the dataset registry.

## Where To Look Next

- [`docs/architecture.md`](architecture.md): current module boundaries
- [`docs/data_organization.md`](data_organization.md): path and run layout overview
- [`docs/run_selectors.md`](run_selectors.md): how to reference existing runs from CLI commands
- [`docs/preprocess.md`](preprocess.md): preprocess flow and concepts
