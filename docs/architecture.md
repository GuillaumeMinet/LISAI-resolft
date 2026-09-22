# LISAI Architecture

This page describes the current production architecture under `src/lisai/**`.

## Main Flows

### Training

1. `lisai train ...` resolves a training config through [`src/lisai/training/cli.py`](../src/lisai/training/cli.py).
2. [`resolve_config`](../src/lisai/config/io/resolver.py) merges and validates configs into `ResolvedExperiment`.
3. [`initialize_runtime`](../src/lisai/training/runtime.py) creates `TrainingRuntime`.
4. [`prepare_data`](../src/lisai/training/setup/data.py) returns `PreparedTrainingData`.
5. [`build_model`](../src/lisai/training/setup/model.py) returns the model and optional restored state.
6. [`get_trainer`](../src/lisai/training/trainers/factory.py) selects the trainer and runs the loop.

### Dataset Registry

1. Preprocessing records the produced dataset structure in the registry after a successful run.
2. [`src/lisai/data/dataset_registry.py`](../src/lisai/data/dataset_registry.py) owns registry loading and dataset metadata helpers.
3. [`src/lisai/data/cli.py`](../src/lisai/data/cli.py) exposes `lisai datasets list/show/open/...` over the same registry.
4. Dataset usage (`training` or `evaluation`) controls the physical dataset root and whether split-based training semantics apply.
5. Training config creation and evaluation can use registry metadata to resolve dataset-dependent fields.

### Config Catalog

1. `lisai configs ...` is registered by [`src/lisai/cli.py`](../src/lisai/cli.py) through [`src/lisai/config/cli.py`](../src/lisai/config/cli.py).
2. [`src/lisai/config/catalog.py`](../src/lisai/config/catalog.py) discovers training configs under `presets/`, `templates/`, `examples/`, and `local/`.
3. Top-level `metadata` is catalog-only and stripped before training validation.
4. Presets and templates are not trainable directly; instantiate them into `configs/training/local/` first.
5. Generated JSON Schemas under `configs/schema/` provide validation/editor metadata for the typed YAML configuration models.

### Inference Configuration

1. [`src/lisai/config/models/inference/**`](../src/lisai/config/models/inference) defines the typed `apply` and `evaluate` configuration contracts.
2. [`src/lisai/evaluation/defaults.py`](../src/lisai/evaluation/defaults.py) resolves sparse inference configs.
3. Resolution precedence is: canonical typed defaults < `configs/inference/local/defaults.yml` < promoted-model config (for `apply --model`) < selected inference config < typed CLI overrides.
4. Apply output routing and input-saving defaults may finally fall back to machine-local settings from `configs/local_config.yml` when the inference config leaves them unspecified.

### Evaluation and Apply

1. [`src/lisai/evaluation/cli.py`](../src/lisai/evaluation/cli.py) resolves model selection and the typed inference config for `lisai apply` and `lisai evaluate`.
2. Run-backed inference uses [`load_saved_run`](../src/lisai/evaluation/saved_run.py) to turn a saved `config_train.yaml` into `SavedTrainingRun`.
3. `apply --model` instead loads a promoted model package and its fixed promoted checkpoint; `evaluate` remains run-based.
4. [`initialize_runtime`](../src/lisai/evaluation/runtime.py) materializes the model/checkpoint into `InferenceRuntime`.
5. [`build_eval_source`](../src/lisai/evaluation/data.py) rebuilds a training split or registered evaluation-only dataset when dataset-based evaluation is needed. [`EvalSource`](../src/lisai/evaluation/source.py) provides the shared semantic identity for training-split versus evaluation-dataset outputs.
6. [`src/lisai/evaluation/inference/**`](../src/lisai/evaluation/inference) performs inference, and [`src/lisai/evaluation/io.py`](../src/lisai/evaluation/io.py) saves predictions, metrics, and output manifests.
7. `evaluate` writes results beneath the source run's `evaluations/` directory; normal `apply` outputs use the configured general inference root unless another output mode is selected.

### Promoted Models

1. `lisai runs promote ...` selects a LISAI training run and packages a chosen state-dict checkpoint into the local reusable model library.
2. [`src/lisai/promoted_models/promotion.py`](../src/lisai/promoted_models/promotion.py) builds the promotion plan and provenance metadata.
3. [`src/lisai/promoted_models/package.py`](../src/lisai/promoted_models/package.py) owns local promoted-model packaging/loading and model-card synchronization.
4. [`src/lisai/promoted_models/registry.py`](../src/lisai/promoted_models/registry.py) tracks locally available promoted models independently of training-run discovery.
5. Export/install and remote catalog/download support live under [`src/lisai/promoted_models/**`](../src/lisai/promoted_models).
6. A promoted model may bundle a sparse inference config and, when required, noise-model artifacts so it can be applied independently of the original training run.

### Preprocess

1. `lisai preprocess ...` is registered by [`src/lisai/cli.py`](../src/lisai/cli.py) through [`src/lisai/preprocess/cli.py`](../src/lisai/preprocess/cli.py).
2. [`run_preprocess_config`](../src/lisai/preprocess/cli.py) loads the YAML config and builds `PreprocessRun`.
3. [`PreprocessRun`](../src/lisai/preprocess/run_preprocess.py) orchestrates preprocessing.
4. Pipelines under [`src/lisai/preprocess/pipelines/**`](../src/lisai/preprocess/pipelines) define dataset transformation logic.
5. The dataset registry is updated after a successful run.

### Runs and External Baselines

1. [`src/lisai/runs/**`](../src/lisai/runs) owns LISAI run metadata, discovery, selectors, lifecycle state, keep/prune operations, plotting, and promotion entrypoints.
2. [`src/lisai/runs/external/**`](../src/lisai/runs/external) defines a separate imported-external-run format used to bring predictions from non-LISAI implementations into the LISAI evaluation layout.
3. `lisai runs list` can display both LISAI and imported external runs, while `lisai runs open --kind external ...` can open an imported external run.
4. External runs preserve provenance/evaluation outputs but are not valid inputs to LISAI `continue`, `apply`, `evaluate`, or promotion workflows.

## Core Boundaries

- `ResolvedExperiment` in [`src/lisai/config/models/training/root.py`](../src/lisai/config/models/training/root.py)
  The typed training config contract.
- Typed inference models in [`src/lisai/config/models/inference/**`](../src/lisai/config/models/inference)
  The `apply`/`evaluate` configuration contracts and sparse override models.
- `TrainingRuntime` in [`src/lisai/training/runtime.py`](../src/lisai/training/runtime.py)
  Live training infrastructure only.
- `PreparedTrainingData` in [`src/lisai/training/setup/data.py`](../src/lisai/training/setup/data.py)
  Typed output of training data setup.
- `TrainingModelSpec` in [`src/lisai/training/setup/model.py`](../src/lisai/training/setup/model.py)
  Training-local model setup contract.
- `SavedTrainingRun` in [`src/lisai/evaluation/saved_run.py`](../src/lisai/evaluation/saved_run.py)
  Evaluation-side view of a saved training run.
- `InferenceRuntime` in [`src/lisai/evaluation/runtime.py`](../src/lisai/evaluation/runtime.py)
  Live inference resources only.
- `EvalSource` in [`src/lisai/evaluation/source.py`](../src/lisai/evaluation/source.py)
  Shared identity for a training split or registered evaluation dataset.
- `Paths` in [`src/lisai/infra/paths/paths.py`](../src/lisai/infra/paths/paths.py)
  Canonical filesystem/path API.
- `PreprocessRun` in [`src/lisai/preprocess/run_preprocess.py`](../src/lisai/preprocess/run_preprocess.py)
  Preprocessing orchestration boundary.
- Promoted-model manifest/registry models in [`src/lisai/promoted_models/schema.py`](../src/lisai/promoted_models/schema.py)
  Portable reusable-model metadata and local-library metadata.
- `ExternalRunMetadata` in [`src/lisai/runs/external/schema.py`](../src/lisai/runs/external/schema.py)
  Metadata contract for imported non-LISAI baselines.

## Directory Responsibilities

- `src/lisai/config/**`: config loading, validation, settings, schema
- `src/lisai/infra/**`: canonical path resolution, filesystem helpers, logging
- `src/lisai/training/**`: training orchestration, setup, trainers, checkpointing
- `src/lisai/models/**`: model architecture registry and load/build logic
- `src/lisai/promoted_models/**`: reusable model promotion, registry, packaging, export/install, and download catalog
- `src/lisai/data/**`: dataset registry/CLI, data loading, data utilities, split manifests, noise-model helpers
- `src/lisai/preprocess/**`: preprocessing configs, pipelines, splitting, saving, and registry updates
- `src/lisai/evaluation/**`: saved-run/promoted-model inference, typed inference-config resolution, evaluation data, inference, metrics, output IO
- `src/lisai/runs/**`: LISAI run metadata/discovery/selectors/lifecycle plus imported external-run support
- `src/lisai/lib/**`: bundled model/library support code used by LISAI models
- `src/scripts/**`: schema generation, legacy/external import, bulk evaluation, and publication/reproducibility helper scripts

## Rule of Thumb

- Config objects describe what should happen.
- Runtime objects carry what is live in the current process.
- Dataset metadata comes from the dataset registry rather than inferred folder conventions when registry information is available.
- LISAI training runs, promoted models, and imported external runs are separate concepts with different lifecycle guarantees.
- Path resolution goes through `Paths`, not hardcoded strings.
