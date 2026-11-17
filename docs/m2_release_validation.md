# LISAI v0.1 — M2 follow-up validation (Patch G)

This checklist verifies the **A → B → C → D → E → F → G** follow-up sequence.
M2 was not released before these changes, so there is no compatibility layer for
intermediate M2 CLI overrides, `local/` directory conventions, or queued jobs.
For a real workstation, back up existing user data/configuration before trying
an explicit workspace update or older three-file global-settings migration.

## 1. Standalone: wheel installed outside the checkout

Install the built wheel in an environment providing the necessary dependencies
(the isolated/minimal dependency audit is deferred to M3). In PowerShell:

```powershell
# No active workspace is required for these read-only operations:
lisai --help
lisai configs list --kind preset
lisai configs list --domain preprocess --kind template

# Explicit creation does not activate a workspace:
lisai workspace init E:/lisai --core-only
lisai workspace init E:/lisai  # add any missing editable config scaffolding
$env:LISAI_HOME = 'E:/lisai'
lisai settings show
lisai configs list
lisai configs list --domain preprocess
lisai configs new denoising_care --output smoke_train
lisai configs new preprocess single_recon --dataset smoke_dataset
lisai configs schema refresh
lisai-runner queue --help
```

Verify `E:/lisai/configs/workspace.yml` and `data.yml` are authoritative;
`training/smoke_train.yml` and `preprocess/smoke_dataset.yml` live **directly**
below their respective config directories; `inference/defaults.yml` and
`inference/post_training.yml` are editable files; editor schemas are under
`configs/schema/`; packaged presets are **not** copied into the workspace.
Generated `CHANGEME` placeholders must be completed before attempting to train
or preprocess real datasets.

With `LISAI_HOME` unset, workspace-dependent commands must fail with a clear
selection error; `settings show`, help, initialization and explicitly scoped
built-in listings must remain usable. A workspace-local `configs/local.yml`
and `LISAI_CONFIG_ROOT` are ignored without warnings or fallback selection.

## 2. Developer checkout and multiple workspaces

In a verified editable LISAI checkout:

```powershell
lisai dev init           # explicitly asks which workspace to select
lisai settings show
lisai settings set --workspace F:/lisai-other --config-mode workspace
lisai settings show
lisai settings set --config-mode custom --config-root D:/my-experiments/configs
lisai settings show
lisai settings set --config-mode repo
lisai settings show
```

Verify that each `settings set` displays the current/proposed selection and
YAML diff and asks for confirmation before writing repository
`configs/local.yml`. Changing W does not implicitly switch an independent
repo/custom R. `config.mode: workspace` follows the selected workspace.
Ordinary commands do **not** accept `--workspace`/`--config-root` overrides.

Also test two distinct W values using explicit `Settings.from_workspace()` in
a single process (A → B → A): dataset and promoted-model registries, run
folders, TensorBoard, inference output, and `.lisai/queue` stay isolated.
`Settings.from_workspace(W, config_root=R)` does not auto-load any `local.yml`.

## 3. Initialization, update and runner checks

```powershell
lisai workspace update             # current selected W
lisai workspace update F:/other    # explicit target; no selection change
lisai workspace update --include-inference
lisai workspace update F:/other --include-inference `
  --external-config-root D:/my-experiments/configs
```

Every update must show **all exact target paths and proposed changes before
confirmation**. Cancelling writes nothing, and unchanged files get no backups.
By default, only the target workspace's global YAMLs are updated. External
inference configs require both explicit options; training/preprocess user
configs are never touched. `--yes` suppresses confirmation, not preview.

Verify full initialization after core-only adds only missing scaffolding and
never changes active selection. For runner jobs, verify the submitted W and R
are retained; the optional **verified repository** developer preference-file
path is read at worker launch. Changing selected W/R later must not redirect a
queued job. No queue migration or broader runner redesign is included.

## 4. Automated verification performed for this patch

On Linux/CPU with Python 3.13, **912 tests were collected and passed** across
four non-overlapping executions (400 configuration/data/scripts; 334
inference/models/runs; 172 training/queue/graphs; 6 top-level CLI tests).
The tests were split to avoid the earlier single-process execution limit.

A `lisai-0.1.0` wheel was built with `pip wheel --no-deps --no-build-isolation`
and installed with `pip install --no-deps --target` into a separate directory.
From `/tmp`, with the wheel as the import source (not the checkout), the smoke
check verified: packaged training presets and preprocess templates, built-in
catalogs without `LISAI_HOME`, missing-workspace error, full workspace init,
`settings show`, flat training and preprocess config creation, schema refresh,
and relative `$schema` links. The source resources were present in the wheel.
The Git ignore rules cover flat/nested user YAMLs and generated schemas, while
package resources stay tracked.

**Not covered:** Windows-specific behavior on the developer workstation,
Python 3.10 runtime testing, isolated dependency installation for a minimal
PyPI environment, end-to-end GPU/HDN/multi-frame training on scientific data,
and ImProcess integration. Those are distinct M3/M4 or workstation acceptance
checks; the scientific algorithms were not intentionally changed by A–G.
