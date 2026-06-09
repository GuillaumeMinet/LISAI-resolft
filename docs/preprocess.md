# Preprocess

This page describes the current preprocessing flow used by LISAI.

## Entry Point

Run preprocessing with either of these commands:

```powershell
lisai preprocess single
lisai preprocess configs/preprocess/single.yml
```

The CLI resolves short config names from `configs/preprocess/` in the same way `lisai train ...` resolves training configs.

Tracked example configs are `single.yml`, `mltpl_snr.yml`, and `timelapse.yml`.

## Flow

1. Load and validate the preprocess YAML config.
2. Build `PreprocessRun` with project paths.
3. Instantiate the requested pipeline.
4. Build the source, output spec, and saver.
5. Optionally plan a train/val/test split.
6. Iterate raw items, transform them, write outputs, and record the preprocess manifest.
7. Update the dataset registry.

## Main Concepts

### `PreprocessRun`

The orchestration layer. It owns runtime configuration, path resolution, pipeline creation, execution, split planning, logging, and registry updates.

### Pipeline

A pipeline defines how a dataset is discovered and transformed. It declares its supported formats, creates a source, processes items, and exposes an output spec.

### Source

The source discovers raw input items and yields them to the pipeline.

### Output Spec

The output spec declares the structure of the generated dataset: output keys, axes, and folder layout.

### Saver

The saver owns file naming, folder creation, and writing processed outputs to disk. When split mode is enabled, it writes directly into final `train/`, `val/`, and `test/` folders.

### Preprocess Manifest

Each preprocess run can write a YAML manifest under the preprocess folder, using the filenames configured in `configs/data_config.yml`. The manifest stores run metadata, source-to-output mappings, and split assignments.

## Split Modes

The preprocess config supports three split modes:

- `random`: deterministic random split using a seed and val/test fractions
- `manual`: assign items explicitly by `source_name`, `source_relpath`, or `sample_id`
- `reuse`: copy assignments from a previous preprocess manifest

Current pipeline names are registered in `src/lisai/preprocess/pipelines/__init__.py`: `single_recon`, `recon_mltpl_snr`, and `recon_timelapse_simple`.

## Outputs

Processed datasets are written under the preprocess area resolved by `Paths.dataset_preprocess_dir(...)`.

If split mode is enabled, the layout matches the existing training loader expectations:

- root outputs: `preprocess/<data_type>/train/...`, `val/...`, `test/...`
- named outputs: `preprocess/<data_type>/<output_key>/train/...`, `val/...`, `test/...`

## Registry

A successful preprocess run updates the dataset registry so the produced dataset can be discovered later by loading and evaluation code. Split summaries are also stored there.
