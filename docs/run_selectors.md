# Run Selectors

Several LISAI commands need to find an existing training run. A run selector is the user-facing way to point those commands at one run folder.

Run selectors are used by:

- `lisai runs open`
- `lisai runs plot`
- `lisai evaluate`
- `lisai apply`
- `lisai continue`

## Start With Runs List

Use `lisai runs list` to inspect available runs:

```powershell
lisai runs list
lisai runs list --dataset Gag
lisai runs list --exp-name upsamp
lisai runs list --run-dir my_model_00
lisai runs list --status running
lisai runs list --full
```

The `run_dir` column is the actual run folder name, for example `my_model_00`. This is the most common selector to copy into other commands.

## Selector Forms

### Stable Run ID

Use `--run-id` when scripting or when you want the most stable reference:

```powershell
lisai runs open --run-id 01ARZ3NDEKTSV4RRFFQ69G7ACD
lisai evaluate --run-id 01ARZ3NDEKTSV4RRFFQ69G7ACD --split val
lisai apply --run-id 01ARZ3NDEKTSV4RRFFQ69G7ACD /data/images
```

`--run-id` cannot be combined with a positional run selector.

### Full Dataset Path

Use `dataset[/subfolder]/run_dir_name` when you know where the run lives:

```powershell
lisai runs plot Gag/Upsamp/my_model_00
lisai evaluate Gag/Upsamp/my_model_00 --metrics psnr,ssim
lisai apply Gag/Upsamp/my_model_00 /data/images
```

If the model subfolder is nested, include all subfolder parts:

```powershell
lisai runs open Gag/Upsamp_base/SubA/my_model_00
```

Do not combine this form with `--dataset`, `--model-subfolder`, or its aliases; the selector already contains that information.

### Run Directory Name

Use the run folder name directly when it is unique enough:

```powershell
lisai runs open my_model_00
lisai runs plot my_model_00
lisai continue my_model_00 --yes
```

You can add filters if the same run folder exists in multiple datasets or subfolders:

```powershell
lisai runs open my_model_00 --dataset Gag --model-subfolder Upsamp
```

`--subfolder` and `--models-subfolder` are accepted aliases for `--model-subfolder`.

### Partial Experiment Name

For interactive use, you can type part of the experiment name:

```powershell
lisai runs list --exp-name reduced
lisai runs plot reduced
lisai evaluate reduced --split val
```

The resolver first checks for an exact run directory name. If no exact folder match exists, it falls back to partial experiment-name matching.

## Ambiguous Matches

If a selector matches multiple runs, LISAI prints a table and asks you to select one when running in an interactive terminal:

```text
Multiple matching runs found:
...
Select run number from '#' (for example 01), or press Enter to cancel:
```

In non-interactive contexts, LISAI cannot ask you to choose. It returns an error and suggests using `--dataset`, `--model-subfolder`/`--subfolder`, or `--run-id`.

## Common Workflows

Inspect and open a run:

```powershell
lisai runs list --exp-name upsamp
lisai runs open my_model_00
```

Plot losses:

```powershell
lisai runs plot my_model_00
```

Evaluate a run on a split:

```powershell
lisai evaluate my_model_00 --split val --metrics psnr,ssim
```

Apply a run to files:

```powershell
lisai apply my_model_00 /data/images --tiling-size 512
```

Continue training in place:

```powershell
lisai continue my_model_00 --yes
```
