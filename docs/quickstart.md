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

LISAI resolves datasets, saved runs, inference outputs, and noise models from a local data root. Create `configs/local_config.yml` on your machine:

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

## Training

Run a public example config from `configs/training/examples`:

```powershell
lisai train examples/vim_denoising_unet
```

You can also pass an explicit file path:

```powershell
lisai train configs/training/examples/vim_denoising_unet.yml
```

Training resolves the config, creates a run directory, saves `config_train.yaml`, and writes checkpoints and logs under the run folder. Example configs assume the referenced datasets already exist under your configured data root.

## Evaluation

Use the CLI for evaluation workflows:

```powershell
lisai evaluate Gag/Upsamp/my_model_00 --split val --metrics psnr,ssim
lisai apply Gag/Upsamp/my_model_00 /data/images --tiling-size 512
lisai apply --run-id 01ARZ3NDEKTSV4RRFFQ69G7ACD /data/images
```

Both commands accept `--config <name>` to load settings from `configs/inference/<name>.yml`,
and any CLI argument overrides the config value. Run selectors must refer to a discovered run
folder, either by `--run-id`, `dataset[/subfolder]/run_dir_name`, `run_dir_name`, or a partial
experiment name when it can be resolved unambiguously.

## Preprocess

Run preprocessing from a YAML config:

```powershell
lisai preprocess single
lisai preprocess configs/preprocess/single.yml
```

Tracked preprocess examples are available in `configs/preprocess/`. They assume the corresponding dataset dump folders already exist under your configured data root.

## Where To Look Next

- [`docs/architecture.md`](architecture.md): current module boundaries
- [`docs/data_organization.md`](data_organization.md): path and run layout overview
- [`docs/run_selectors.md`](run_selectors.md): how to reference existing runs from CLI commands
- [`docs/preprocess.md`](preprocess.md): preprocess flow and concepts
