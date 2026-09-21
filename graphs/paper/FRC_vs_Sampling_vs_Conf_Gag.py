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

from lisai.data.utils import crop_center
from graphs.utils.eval_folder import EvalSource, discover_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir


# data selection
dataset_name = "gag_live_tl_highON"
model_subfolder = "Upsamp"
run_names = [
    "CL1_Upsamp05_biggerNet_01",
    "CL1_Upsamp2_biggerNet_02"
]

evaluation_source = EvalSource.dataset("gag_live_upsamp_eval")
checkpoint = "best"

crop_size = None
smooth_gt = True

# figure parameters
show_figure = True
include_legend=False
colors_list = ["black", "grey", "#a70048ff", "#0000ffff"]
linewidth = 0.4
figure_size = (1, 1)
fontsize = 5.5

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Upsamp_FRC_vs_Sampling_vs_Conf_Gag.svg"


# find and align evaluations
evaluations = discover_evaluations(
    dataset=dataset_name, 
    model_subfolder=model_subfolder, 
    run_names=run_names,
    source=evaluation_source,
    checkpoint=checkpoint
)

if any(evaluation.sampling_ratio is None for evaluation in evaluations):
    for evaluation in evaluations:
        print(evaluation.name, evaluation.sampling_ratio)
    raise ValueError("Could not determine sampling ratio for one or more runs.")

evaluations.sort(key=lambda evaluation: evaluation.sampling_ratio)

run_conditions = [f"S={evaluation.sampling_ratio:g}" for evaluation in evaluations]

conditions = ["GT", "Confocal"] + run_conditions

comparison = prepare_comparison_outputs(evaluations)

frc_curves = {cond: [] for cond in conditions}
avg_frc_curves = {cond: [] for cond in conditions}
std_frc_curves = {cond: [] for cond in conditions}


def add_frc(arr, cond):
    if crop_size is not None: 
        arr = crop_center(arr, crop_size)
    if cond == "GT" and smooth_gt: 
        arr = gaussian_filter(arr, sigma=0.3, radius=3)
    if cond in ("GT","Confocal"): 
        arr[arr < 0] = 0
    arr = (arr - np.mean(arr)) / np.std(arr)
    arr = arr - np.min(arr)
    arr = frc.util.apply_tukey(arr)
    frc_curve = frc.one_frc(arr)
    if np.isnan(frc_curve).any():
        print(f"Skipping image in {cond} due to NaN values in FRC curve.")
        return False
    frc_curves[cond].append((frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve)))
    return True


# GT and confocal are dataset data, identical across aligned runs: load once
count_gt = 0
count_conf = 0
img_size = None
for item in comparison.items_for(evaluations[0]):
    gt_frames, _ = item.load_gt_pred()
    conf_frames = item.load_aux_data("conf")
    if len(gt_frames) != len(conf_frames): 
        raise ValueError("GT and confocal sample counts do not match.")
    for gt, conf in zip(gt_frames, conf_frames):
        img_size = crop_center(gt, crop_size).shape if crop_size is not None else gt.shape
        count_gt += int(add_frc(gt, "GT"))
        count_conf += int(add_frc(conf, "Confocal"))
print(f"GT: #{count_gt} files")
print(f"Confocal: #{count_conf} files")


# predictions
for evaluation, cond in zip(evaluations, run_conditions):
    count = 0
    print(f"\n{cond}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")
    for item in comparison.items_for(evaluation):
        _, pred_frames = item.load_gt_pred()
        for pred in pred_frames: count += int(add_frc(pred, cond))
    print(f"{cond}: #{count} files")


# average and standard deviation
for cond in conditions:
    frc_curves[cond] = np.stack(frc_curves[cond])
    avg_frc_curves[cond] = np.mean(frc_curves[cond], axis=0)
    std_frc_curves[cond] = np.std(frc_curves[cond], axis=0)


# plot
scale = 1 / 30 * 1e3
fig, ax = plt.subplots(1, 1, figsize=figure_size)
for cond, color in zip(conditions, colors_list):
    frc_curve = avg_frc_curves[cond]
    std_curve = std_frc_curves[cond]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve, label=cond, linewidth=linewidth, color=color)
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.3, color=color, linewidth=0)
if include_legend:
    plt.legend()

ticks_prms = {"labelsize": fontsize, "width": 0.7, "length": 2}
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
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

if show_figure: 
    plt.show()
if save_figure: 
    fig.savefig(save_folder / save_title, bbox_inches="tight")
