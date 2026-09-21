import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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

run_names = [
    "SnrHigh_unpaired_S025",
    "SnrHigh_unpaired_S050",
    "SnrHigh_unpaired_S075",
]

evaluation_source = EvalSource.training_split("test")
checkpoint = "best"  # "best" or "last"


# FRC parameters
scale = 1 / 30 * 1e3  # um-1


# figure parameters
show_figure = True

fontsize = 5.5
linewidth = 0.7
colors = ["#a70048ff", "#0000ffff", "#439c43fb"]
xticks_positions = [0, 5, 10, 15]
yticks_positions = [0, 0.5, 1]
ticks_prms = {"labelsize": fontsize, "width": linewidth, "length": 2}


# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Upsamp_relativeFRC_HighSNR_vsSampling.svg"


# find evaluations
evaluations = discover_evaluations(
    dataset=dataset_name,
    model_subfolder=model_subfolder,
    run_names=run_names,
    source=evaluation_source,
    checkpoint=checkpoint,
)

if any(evaluation.sampling_ratio is None for evaluation in evaluations):
    raise ValueError("Could not determine the sampling ratio for one or more runs.")

evaluations.sort(key=lambda evaluation: evaluation.sampling_ratio)

sampling_ratios = [evaluation.sampling_ratio for evaluation in evaluations]
sampling_labels = [f"S={sampling:g}" for sampling in sampling_ratios]


# exact same source images across sampling ratios
comparison = prepare_comparison_outputs(evaluations, gt_reference="dataset")


# initialize FRC storage
frc_curves = {label: [] for label in sampling_labels}
img_size = None


# relative FRC: GT vs prediction
for evaluation, label in zip(evaluations, sampling_labels):
    count = 0

    print(f"\n{label}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")

    for item in comparison.items_for(evaluation):
        gt_frames, pred_frames = item.load_gt_pred()

        for gt, pred in zip(gt_frames, pred_frames):
            count += 1
            print(f"Relative FRC {count} for {label}")
            gt = gaussian_filter(gt, sigma=0.6, radius=3)

            gt[gt<-3] = -3

            gt = (gt - np.mean(gt)) / np.std(gt)
            pred = (pred - np.mean(pred)) / np.std(pred)

            gt = gt - np.min(gt)
            pred = pred - np.min(pred)
            gt = frc.util.apply_tukey(gt)
            pred = frc.util.apply_tukey(pred)

            frc_curve = frc.two_frc(gt, pred)
            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))

            frc_curves[label].append(frc_curve)
            img_size = gt.shape[0]


# average and standard deviation
avg_frc_curves = {}
std_frc_curves = {}

for label in sampling_labels:
    curves = np.stack(frc_curves[label])
    avg_frc_curves[label] = np.mean(curves, axis=0)
    std_frc_curves[label] = np.std(curves, axis=0)


# plot
fig, ax = plt.subplots(1, 1, figsize=(1, 1))

for label, color in zip(sampling_labels, colors):
    frc_curve = avg_frc_curves[label]
    std_curve = std_frc_curves[label]

    xs_pix = np.arange(len(frc_curve)) / img_size
    xs_nm_freq = xs_pix * scale

    ax.plot(xs_nm_freq, frc_curve, label=label, linewidth=0.2, color=color)
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve,
                    color=color, alpha=0.5, linewidth=0)


ax.set_xticks(xticks_positions)
ax.set_yticks(yticks_positions)

ax.tick_params(axis="x", which="major", **ticks_prms)
ax.tick_params(axis="y", which="major", **ticks_prms)

ax.set_xlabel("Spatial frequency (µm$^{-1}$)", fontsize=fontsize)
ax.set_ylabel("Correlation", fontsize=fontsize)

ax.set_ylim(0, 1)
ax.set_xlim(0, 15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(linewidth)
ax.spines["bottom"].set_linewidth(linewidth)


# show
if show_figure:
    plt.show()


# save
if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches="tight")