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

## Dataset Usage

`usage: training` datasets may use the train/validation/test split machinery below.

`usage: evaluation` datasets are whole-dataset evaluation resources. Splitting is disabled for them: outputs are written directly under `preprocess/<data_type>/...`, manifest items record no split, and the registry keeps only `split.enabled: false` rather than fabricating a training split.

## Split Modes

The preprocess config supports three split modes for training datasets:

- `random`: deterministic random split using a seed and val/test fractions
- `manual`: assign items explicitly by `source_name`, `source_relpath`, or `sample_id`
- `reuse`: copy assignments from a previous preprocess manifest

Current pipeline names are registered in `src/lisai/preprocess/pipelines/__init__.py`: `single_recon`, `paired_single_recon`, `recon_mltpl_snr`, and `recon_timelapse_simple`.

### Source subfolders

New preprocess configs use `base_subfolder` for an optional common parent under `dump/<data_type>/`, and `input_subfolder` / `gt_subfolder` / `auxiliary_subfolders` for role-specific collections beneath that base. For example, `base_subfolder: gag` with `input_subfolder: resolft` resolves to `dump/recon/gag/resolft`.

`dump_subfolder` remains a deprecated compatibility alias. Its legacy meaning is preserved per pipeline: for `single_recon` it maps to `input_subfolder`; for `paired_single_recon`, `recon_timelapse_simple`, and `recon_mltpl_snr` it maps to `base_subfolder`. Supplying both the legacy field and its new equivalent is an error.

## Outputs

Processed datasets are written under the preprocess area resolved by `Paths.dataset_preprocess_dir(...)`.

For pipelines with a direct source-stream mapping, role-specific source names are preserved in the processed layout. `single_recon` mirrors a new `input_subfolder` by default, while `paired_single_recon` writes its input and GT under their configured `input_subfolder` and `gt_subfolder` names. `base_subfolder` is source organization only and is not mirrored into `preprocess/`.

For `single_recon`, `output_subfolder` can override the primary processed destination. Omitting it mirrors `input_subfolder`; setting it explicitly to `null` saves the primary images directly at `preprocess/<data_type>/`. Legacy `dump_subfolder` configs retain the historical root-output behavior.

If split mode is enabled, split folders are appended beneath the resolved output path, e.g. `preprocess/<data_type>/<output_subfolder>/train/...`, `val/...`, and `test/...`.

## Registry

A successful preprocess run updates the dataset registry so the produced dataset can be discovered later by loading and evaluation code. Split summaries are also stored there.

### Auxiliary matching

`single_recon` and `paired_single_recon` can attach named auxiliary image folders. Each auxiliary must explicitly declare whether matching is required or optional. Exact filename matching remains the default:

```yaml
auxiliary_subfolders:
  conf:
    matching: required
```

Use `matching: optional` to keep primary samples that have no auxiliary counterpart. To match acquisitions whose filenames contain timestamps such as `00h22m40s`, use one-to-one timestamp matching with an explicit maximum time difference:

```yaml
auxiliary_subfolders:
  conf:
    matching: required
    match_by: timestamp
    max_time_delta_s: 30
    timestamp_relation: before_primary  # before_primary | after_primary | either
```

`timestamp_relation` defaults to `either`. `max_time_delta_s` is mandatory for timestamp matching.
