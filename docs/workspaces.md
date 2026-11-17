# LISAI v0.1 — Workspaces and developer setup

A **workspace** owns persistent data, model bundles, queues and authoritative
`configs/workspace.yml` and `configs/data.yml` settings. The separate **writable
experiment-config root** contains editable training, preprocessing and inference
YAML files. Immutable presets, templates and examples live in the LISAI package.

## Standalone CLI

```sh
lisai workspace init /absolute/path/to/LISAI
export LISAI_HOME=/absolute/path/to/LISAI
lisai settings show
lisai configs list
```

On PowerShell, set the variable for the current terminal with
`$env:LISAI_HOME = 'D:/LISAI'`. The standalone CLI requires `LISAI_HOME` for
workspace-dependent commands; it does **not** choose `~/LISAI` automatically.
Initializing a workspace does not select it or change any environment variable.

Open the currently selected workspace in the native file explorer with
`lisai workspace open` (Windows Explorer, Finder or your Linux file manager).
This command is read-only and works for developer and standalone selection.

For a workspace that will use experiment files in a developer repository, only
the workspace-global infrastructure may be needed:

```sh
lisai workspace init /absolute/path/to/LISAI --core-only
```

**Core-only** creates missing global `workspace.yml` and `data.yml` files plus
`datasets/`, `models/`, `noise_models/`, `inference/`, `tensorboard/`, and
`.lisai/queue/`. It does *not* create experiment-configuration folders, inference
YAMLs or generated editor schemas. Such a workspace is valid: operations require
only the files they actually use.

The default (full) `workspace init PATH` adds missing `configs/training/`,
`preprocess/`, `inference/defaults.yml`, `inference/post_training.yml` and
`schema/` under the named workspace. You can run full initialization after
core-only initialization. Both operations are idempotent and preserve existing
YAMLs, datasets, checkpoints, models and other user files.

## Editable developer checkout

In a verified editable LISAI checkout, initialize the repository and then
select a workspace **separately**:

```sh
lisai dev init
lisai workspace init /data/lisai --core-only    # only if needed
lisai settings set --workspace /data/lisai
lisai settings show
lisai configs schema refresh
```

`dev init` never asks for or selects a workspace, and does not initialize one.
It creates missing repo-owned experiment folders, seeds editable inference
`defaults.yml` and `post_training.yml`, and writes a developer `configs/local.yml`
with `dev.enabled: true` and default `config.mode: repo` if none exists.
An already selected workspace/config mode is preserved when running it again.

**Developer setup is complete** when developer mode is enabled and both
`workspace.root` and `config.mode` are valid. A fresh `dev init` deliberately
leaves `workspace.root` unset and displays the `lisai settings set --workspace
PATH` command. Workspace-dependent CLI commands fail with this instruction
until selection is complete, even if `LISAI_HOME` happens to be defined.
The selected workspace must separately contain valid `configs/workspace.yml`
and `configs/data.yml`; create them with `workspace init PATH --core-only` if
missing. Read-only `settings show`, `settings set`, explicit workspace setup and
packaged catalog commands remain available during incomplete setup.

`dev init` can generate workspace-independent editor schemas before selection.
Run `lisai configs schema refresh` after selecting a workspace to generate the
registry-aware experiment/preprocess schemas. Initialization never modifies the
shell environment or rewrites user-authored configurations.

A developer can choose the persisted experiment-config ownership:

```sh
lisai settings set --config-mode repo
lisai settings set --workspace /data/test --config-mode workspace
lisai settings set --config-mode custom --config-root /data/custom-configs
```

`repo` means the editable checkout's `configs/`; `workspace` means the selected
workspace's `configs/`; `custom` means the explicitly specified path.
`settings set` previews and confirms changes to repository `configs/local.yml`.
Ordinary CLI commands have no live `--workspace` or `--config-root` override.
The optional `LISAI_CONFIG_ROOT` environment variable is ignored. A standalone
workspace-local `configs/local.yml` is also ignored.

Editable experiment configurations have a flat layout, for example
`<config-root>/training/my_hdn.yml`, `<config-root>/preprocess/vim.yml` and
`<config-root>/inference/defaults.yml`. `local/` may be used as an ordinary nested
folder name, but is not required. Repo-owned editable configs and schemas are
Git-ignored; packaged presets/templates/examples are tracked in package resources.

If your editor reports that `config.mode` is disallowed in developer
`configs/local.yml`, regenerate the editor schemas with
`lisai configs schema refresh`. This also refreshes the checkout-owned
`configs/schema/local-config.schema.json` when `config.mode` is `workspace`
or `custom`, even though the writable experiment root is elsewhere.

## Updating workspace defaults

```sh
lisai workspace update                          # selected workspace
lisai workspace update /data/other-workspace    # explicit target; no selection change
lisai workspace update --include-inference
lisai workspace update /data/other-workspace --include-inference \
  --external-config-root /data/custom-configs
```

**Every update lists exact paths and proposed diffs before modifying files.**
It then requests confirmation. `--yes` skips only confirmation, not the preview.
Cancellation and no-op updates create no backups or writes. Changed existing
files receive a unique backup, and replacements are written atomically per file.

By default `workspace update` affects **only the target workspace's**
`configs/workspace.yml` and `configs/data.yml`. With `--include-inference`, it
also refreshes that same workspace's `configs/inference/defaults.yml` and
`post_training.yml`. It never infers an external experiment-config root from
active developer selection. Updating an external root requires *both*
`--include-inference` and `--external-config-root PATH`, and the preview clearly
labels the external target. Training/preprocessing configurations, datasets,
models and runs are never changed by workspace updates. JSON-schema regeneration
is a separate explicit operation (`lisai configs schema refresh`).

An outdated `defaults_revision` prints a warning during normal CLI execution;
settings loading never automatically updates files.

## Older three-file global settings migration

The optional `workspace migrate` command only converts older M1-style global
configuration files. It does **not** migrate an unreleased M2 layout:

```sh
lisai workspace migrate /absolute/old/data/root \
  --from-legacy /path/to/old/checkout/configs
lisai workspace init /absolute/old/data/root
```

You may omit the destination when an active workspace has already been selected.
The old `project_config.yml` and `data_config.yml` are no longer shipped in the
repository `configs/` directory; historical test copies live under
`tests/fixtures/legacy_configs/`. The source directory is always required.
Migration validates that the old
`infrastructure.data_root` matches the destination, converts path templates to
`{workspace_root}`, and creates workspace-global YAMLs without overwriting
existing ones or moving the user's data. To add the experiment scaffolding,
run full `workspace init` explicitly afterward.

## Queue worker

Submitted jobs carry the resolved workspace root and writable config root. For
an activated developer checkout they also carry the explicit repository
`configs/local.yml` path for permitted preferences. The worker reads those
preferences at launch but never uses that file to reselect its submitted
workspace or config root. Later edits to preferences may therefore affect an
unstarted job; changing the active workspace cannot redirect it. Standalone
jobs never read workspace-local `local.yml`. No queued-job migration or runner
scheduling redesign is included in M2.
