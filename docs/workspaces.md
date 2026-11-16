# LISAI v0.1: workspaces and developer setup

Persistent data belongs to a **workspace**, not the LISAI package or source checkout.
A workspace has authoritative `configs/workspace.yml` and `configs/data.yml`, plus
`datasets/`, `models/`, `noise_models/`, `inference/`, `tensorboard/` and `.lisai/queue/`.
Dataset training runs remain inside each dataset's `runs/` directory.

## Standalone

```sh
lisai workspace init /absolute/path/to/LISAI
lisai --workspace /absolute/path/to/LISAI config show
lisai --workspace /absolute/path/to/LISAI configs list
```

`workspace init [PATH]` is **explicit** and idempotent. It creates missing global
settings, directories, local inference defaults, editor schemas and empty local
config directories. It never overwrites existing YAMLs, models, datasets or
runs. After setup, edit experiment YAMLs under `configs/training/local/` and
`configs/preprocess/local/`; package presets/templates stay read-only.

The default workspace is `LISAI_HOME` (absolute path) or `~/LISAI`.

## Editable checkout

From `pip install -e '.[dev]'` in a verified source checkout:

```sh
lisai --workspace /absolute/path/to/LISAI dev init
lisai config show
```

`dev init` asks before creating an absent workspace core. For unattended setup,
pass `--yes` to consent. It initializes the workspace **core only**, and writes
`configs/local.yml`, editable inference defaults and local experiment folders
into the **repo**. The local file is gitignored and contains only the workspace
selector and developer activation flag. No environment variable is modified.
An already existing conflicting `local.yml` is never overwritten; edit it
explicitly instead.

With developer activation, `--workspace X` changes persistent settings and data
but retains the repo experiment-config root. To use X's configs as well, add
`--config-root X/configs` (or set `LISAI_CONFIG_ROOT` to an absolute path).

## Updating defaults

```sh
lisai --workspace /absolute/path/to/LISAI workspace update
lisai --workspace /absolute/path/to/LISAI workspace update --include-inference
```

Settings reads, imports and ordinary commands never auto-update. An outdated
`defaults_revision` in `workspace.yml` causes a warning. `workspace update`
requires confirmation (or explicit `--yes`), backs up the existing
`workspace.yml` and `data.yml` with timestamped `.bak` filenames, and installs
current packaged defaults. The `--include-inference` flag additionally backs up
and replaces the editable inference defaults in the **selected config root**.
Never use update to edit user training/preprocess/experiment files or registries.
`lisai configs schema refresh` explicitly refreshes editor schema caches.

## Legacy M1 migration

```sh
lisai --workspace /absolute/old/data/root workspace migrate \
  --from-legacy /path/to/old/checkout/configs
lisai --workspace /absolute/old/data/root workspace init
```

The first command reads the legacy `local_config.yml`, `project_config.yml` and
`data_config.yml` and creates the new workspace-global files. It requires the
legacy `infrastructure.data_root` to equal the selected workspace root. It
converts old path templates to `{workspace_root}`, preserving run paths, and
retains old inference/console preferences as workspace defaults. The original
YAMLs, weights and data are not altered, deleted, or moved. It refuses to
overwrite an existing `workspace.yml` or `data.yml`.

After migration, explicitly run `workspace init` to add any missing directories,
local inference configs and editor schemas, and `dev init` separately if using
source-checkout-local experiment configs. Developer `local.yml` is deliberately
not auto-generated from legacy settings: it is an independent, opt-in selection.
