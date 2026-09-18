import sys
import random
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

from lisai.data.utils import extract_patches
from graphs.utils.eval_folder import EvalSource, discover_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir


# data selection
dataset_name = "mito_live_tl_highON"
model_subfolder = "Upsamp"

run_names = [
    "CL1_S025",
    "CL5_S025",
]

evaluation_source = EvalSource.training_split("test")
checkpoint = "last"  # "best" or "last"


# FRC calculation parameters
n_imgs = 100
random_imgs = True
patch_size = 300

smooth_gt = True
plot_gt = True


# figure parameters
show_figure = True

colors_list = ["mediumblue", "#0d9188ff"]
linewidth = 0.7
figure_size=(1.4, 1.4)
labels_fontSize = 8
labelsize = 8

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Upsamp_FRC_vs_ContextLength_Mito.svg"


# find evaluations
evaluations = discover_evaluations(
    dataset=dataset_name,
    model_subfolder=model_subfolder,
    run_names=run_names,
    source=evaluation_source,
    checkpoint=checkpoint,
)

evaluations.sort(key=lambda evaluation: evaluation.context_length)

cl_list = [f"N={evaluation.context_length}" for evaluation in evaluations]


# exact source-item and timepoint alignment across runs
comparison = prepare_comparison_outputs(evaluations)


# select same aligned images for all conditions
sample_counts = [
    sum(len(item.positions) for item in comparison.items_for(evaluation))
    for evaluation in evaluations
]

if len(set(sample_counts)) != 1:
    raise ValueError(f"Aligned evaluations have inconsistent sample counts: {sample_counts}")

n_available = sample_counts[0]

if n_imgs is not None and n_imgs < n_available:
    if random_imgs:
        selected_idxs = set(random.sample(range(n_available), n_imgs))
    else:
        selected_idxs = set(range(n_imgs))
else:
    selected_idxs = set(range(n_available))

print(f"Using {len(selected_idxs)}/{n_available} aligned images")


# initialize FRC storage
frc_curves = {"gt": []}
for cl in cl_list:
    frc_curves[cl] = []


# GT calculation
# GT is identical across aligned evaluations, so calculate it only once
count = 0
global_idx = 0
img_size = None

for item in comparison.items_for(evaluations[0]):
    gt_frames = item.load_samples("gt", squeeze=True)

    for gt in gt_frames:
        if global_idx not in selected_idxs:
            global_idx += 1
            continue

        count += 1
        global_idx += 1

        print(f"GT {count}/{len(selected_idxs)}")

        img_size = gt.shape

        if smooth_gt:
            gt = gaussian_filter(gt, sigma=0.4, radius=3)

        gt = (gt - np.mean(gt)) / np.std(gt)
        gt = gt - np.min(gt)

        patches_gt = extract_patches(gt, patch_size)

        for patch_idx in range(patches_gt.shape[0]):
            patch_gt = frc.util.apply_tukey(patches_gt[patch_idx])
            frc_curve = frc.one_frc(patch_gt)

            if np.isnan(frc_curve).any():
                print(f"Skipping patch {patch_idx} in GT due to NaN values in FRC curve.")
                continue

            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
            frc_curves["gt"].append(frc_curve)


# prediction FRC
for evaluation, cl in zip(evaluations, cl_list):
    count = 0
    global_idx = 0

    print(f"\n{cl}: {evaluation.run.name}")
    print(f"  {evaluation.folder}")

    for item in comparison.items_for(evaluation):
        pred_frames = item.load_samples("pred", squeeze=True)

        for pred in pred_frames:
            if global_idx not in selected_idxs:
                global_idx += 1
                continue

            count += 1
            global_idx += 1

            print(f"Prediction {count}/{len(selected_idxs)} for {cl}")

            pred = (pred - np.mean(pred)) / np.std(pred)
            pred = pred - np.min(pred)

            patches = extract_patches(pred, patch_size)

            for patch_idx in range(patches.shape[0]):
                patch = frc.util.apply_tukey(patches[patch_idx])
                frc_curve = frc.one_frc(patch)

                if np.isnan(frc_curve).any():
                    print(f"Skipping patch {patch_idx} in {cl} due to NaN values in FRC curve.")
                    continue

                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves[cl].append(frc_curve)


# average and standard deviation
avg_frc_curves = {}
std_frc_curves = {}

for key in ["gt"] + cl_list:
    curves = np.stack(frc_curves[key])
    avg_frc_curves[key] = np.mean(curves, axis=0)
    std_frc_curves[key] = np.std(curves, axis=0)


# plot
scale = 1 / 34 * 1e3

fig, ax = plt.subplots(1, 1, figsize=figure_size)


# GT
if plot_gt:
    frc_curve = avg_frc_curves["gt"]
    std_curve = std_frc_curves["gt"]

    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale

    ax.plot(xs_nm_freq, frc_curve, label="GT", linewidth=linewidth * 0.6, color="black")
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve,
                    alpha=0.2, color="black", linewidth=0)


# predictions
for cl, color in zip(cl_list, colors_list):
    frc_curve = avg_frc_curves[cl]
    std_curve = std_frc_curves[cl]

    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale

    ax.plot(xs_nm_freq, frc_curve, label=cl, linewidth=linewidth, color=color)
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve,
                    alpha=0.3, color=color, linewidth=0)



ticks_prms = {"labelsize": labelsize, "width": linewidth, "length": 3 * linewidth}

ax.set_xlabel("Spatial frequency (µm$^{-1}$)", fontsize=labels_fontSize)
ax.set_ylabel("Correlation", fontsize=labels_fontSize)

ax.set_xlim(0, 14)
ax.set_ylim(0, 1)

ax.set_xticks([0, 5, 10])
ax.set_yticks([0, 0.5, 1])

ax.tick_params(axis="x", which="major", **ticks_prms)
ax.tick_params(axis="y", which="major", **ticks_prms)

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