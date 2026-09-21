import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import frc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from graphs.utils.eval_folder import EvalSource, discover_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir


# data selection
dataset_name = "vim_fixed_multi_snr"
model_subfolder = "Upsamp"
run_specs = [
    ("SnrHigh_unpaired_S025", 0.25),
    ("SnrHigh_unpaired_S050", 0.50),
    ("SnrHigh_unpaired_S075", 0.75),
]
evaluation_source = EvalSource.training_split("test")
checkpoint = "best"

smooth_gt = True

# figure parameters
show_figure = True
colors_per_cond = {"GT": "black", "S=0.25": "#a70048ff", "S=0.5": "#0000ffff", "S=0.75": "#439c43fb"}
linewidth = 0.7
figure_size = (1.4, 1.4)
fontsize = 8

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Upsamp_1FRC_Sampling_highSNRonly_vim.svg"


# discover and align evaluations
run_names = [name for name, _ in run_specs]
sampling_ratios = [sampling for _, sampling in run_specs]
evaluations = discover_evaluations(dataset=dataset_name, model_subfolder=model_subfolder, run_names=run_names,
                                   source=evaluation_source, checkpoint=checkpoint)
conditions = ["GT"] + [f"S={sampling:g}" for sampling in sampling_ratios]
evaluation_by_condition = {f"S={sampling:g}": evaluation for evaluation, sampling in zip(evaluations, sampling_ratios)}
comparison = prepare_comparison_outputs(evaluations, required=("pred",), gt_reference="dataset")

frc_curves = {cond: [] for cond in conditions}
img_size = None


def add_frc(arr, cond):
    if cond == "GT" and smooth_gt: arr = gaussian_filter(arr, sigma=0.6, radius=3)
    if cond == "GT": arr[arr < 0] = 0
    arr = (arr - np.mean(arr)) / np.std(arr)
    arr = arr - np.min(arr)
    arr = frc.util.apply_tukey(arr)
    frc_curve = frc.one_frc(arr)
    if np.isnan(frc_curve).any():
        print(f"Skipping image in {cond} due to NaN values in FRC curve.")
        return False
    frc_curves[cond].append((frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve)))
    return True


# one canonical dataset GT, shared across all aligned runs
count = 0
for item in comparison.items_for(evaluations[0]):
    gt_frames = comparison.load_reference_gt(item)
    for gt in gt_frames:
        img_size = gt.shape
        count += int(add_frc(gt, "GT"))
print(f"GT: #{count} files")


# predictions
for cond in conditions[1:]:
    evaluation = evaluation_by_condition[cond]
    count = 0
    print(f"\n{cond}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")
    for item in comparison.items_for(evaluation):
        for pred in item.load_samples("pred", squeeze=True): count += int(add_frc(pred, cond))
    print(f"{cond}: #{count} files")


# average and standard deviation
avg_frc_curves = {}
std_frc_curves = {}
for cond in conditions:
    curves = np.stack(frc_curves[cond])
    avg_frc_curves[cond] = np.mean(curves, axis=0)
    std_frc_curves[cond] = np.std(curves, axis=0)


# plot
scale = 1 / 30 * 1e3
fig, ax = plt.subplots(1, 1, figsize=figure_size)
for cond in conditions:
    color = colors_per_cond[cond]
    frc_curve = avg_frc_curves[cond]
    std_curve = std_frc_curves[cond]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve, label=cond, linewidth=linewidth, color=color)
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve,
                    alpha=0.3, color=color, linewidth=0)


ticks_prms = {"labelsize": fontsize, "width": linewidth, "length": 3 * linewidth}
ax.set_xlabel("Spatial frequency (µm$^{-1}$)", fontsize=fontsize)
ax.set_ylabel("Correlation", fontsize=fontsize)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.set_xticks([0, 5, 10, 15])
ax.set_yticks([0, 0.5, 1])
ax.tick_params(axis="x", which="major", **ticks_prms)
ax.tick_params(axis="y", which="major", **ticks_prms)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(linewidth)
ax.spines["bottom"].set_linewidth(linewidth)
ax.legend()

if show_figure:
    plt.show()
if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches="tight")
