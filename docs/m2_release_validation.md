# LISAI v0.1 — M2 validation checklist

Patch G closes the configuration/workspace milestone. The following checks are
recommended on the Windows developer machine **after applying A–G in order**.
Use a backup of any existing `configs/` files and dataset registries when
trying migration or `workspace update` for the first time.

## Clean standalone installation

Install the built wheel into a Python environment that already provides the
required dependencies (full dependency validation is M3), then run from a
folder **outside** the checkout:

```powershell
lisai workspace init E:/lisai
lisai --workspace E:/lisai config show
lisai --workspace E:/lisai configs list
lisai --workspace E:/lisai configs list --domain preprocess
lisai --workspace E:/lisai configs new denoising_care --name smoke_train
lisai --workspace E:/lisai configs new preprocess single_recon --dataset smoke_dataset
lisai --workspace E:/lisai configs schema refresh
lisai-runner --workspace E:/lisai queue --help
```

Check that global YAMLs are in `E:/lisai/configs/`, generated local
experiment configs are in `E:/lisai/configs/{training,preprocess}/local/`, and
built-in presets do **not** get copied there. Edit generated `CHANGEME` values
and dataset-specific source fields before executing training/preprocessing.

## Developer installation and two workspaces

From a verified editable LISAI checkout:

```powershell
lisai --workspace E:/lisai dev init
lisai config show
lisai --workspace F:/lisai-other config show
lisai --workspace F:/lisai-other --config-root F:/lisai-other/configs config show
```

Verify that the first two commands use the checkout's `configs/` for editable
experiments while changing the workspace/data root as requested. The third
switches the experiment-config root explicitly to the other workspace.
Check that the per-workspace dataset/model registries, run folders,
TensorBoard directories, inference results and `.lisai/queue` directories
remain independent. Compare A→B→A in a single Python process if testing APIs.

## Existing installations

The previous three-file configuration is not loaded automatically by M2.
For an existing `E:/lisai` data root, consult [workspaces.md](workspaces.md)
for explicit non-destructive `workspace migrate`, followed by `workspace init`
for missing scaffolding. Do not run `workspace update` without first reviewing
its backups and replacement behavior.

## Release-gate evidence and scope

The Patch G automated Linux/CPU validation ran all 874 collected tests in
three non-overlapping groups (352 configuration/data, 327 inference/models/runs,
195 training/queue/graph/misc), each passing. The all-in-one command did not
finish within the execution limit; the grouped runs cover the same tests.
A wheel was installed and exercised from outside the source checkout, with the
pre-existing Python dependencies shared from the testing environment.

**Not certified by M2:** GPU training on the real workstation, full HDN and
multi-frame prediction end-to-end, isolated dependency resolution for a minimal
PyPI install, and the ImProcess integration. These remain M3/M4 or user
acceptance checks; the scientific algorithms were not changed in M2.
