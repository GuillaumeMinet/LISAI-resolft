import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot as new_box_plot
from graphs.utils.run_selection import discover_graph_runs
from graphs.utils.eval_folder import EvalSource, resolve_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir

# since this script is focused on unsupervised HDN,
# metrics evaluation can be done on full dataset: training + val + test
# put False to use only test dataset
full_dataset = True

# data selection
dataset_name = "vim_fixed_multi_snr"
model_subfolder = "HDN_benchmark"

run_names = [
    "vim_single_betaKL03_01",
    "vim_single_betaKL05_04",
    "vim_single_betaKL07_01",
]

checkpoint = "best"  # "best" or "last"

if full_dataset:
    evaluation_sources = [
        EvalSource.training_split("train"),
        EvalSource.training_split("val"),
        EvalSource.training_split("test"),
    ]
else:
    evaluation_sources = [EvalSource.training_split("test")]


# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Denoising_Metrics_vs_BetaKL_Vim.svg"


# metrics calculation parameters
smooth_gt = True
use_windowed = False
window_size = 600
range_invariant = False
patch_selection = 0.3


# figure parameters
show_figure = True

figsize = (10, 5)
spaceBetweenSubplots = 0.4

colors_list = ["#2ee2f0ff", "#52c2f3ff", "#0a6f9cff"]

box_plot_parameters = {
    "colors": colors_list,
    "widths": 0.1,
    "positions": [0.1, 0.25, 0.4],
    "linewidth": 2,
    "dashed_whiskers": True,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": True,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.7,
    "dataPoints_color": "same",
    "xlabel": r"$\beta_{\mathrm{KL}}$",
    "use_mean": True,
    "labels_fontSize": 20,
    "ticks_prms": {"labelsize": 20, "width": 2, "length": 8},
}


# find runs
runs = discover_graph_runs(
    dataset=dataset_name,
    model_subfolder=model_subfolder,
    run_names=run_names,
)

if any(run.beta_kl is None for run in runs):
    raise ValueError("Could not determine betaKL for one or more runs.")

runs.sort(key=lambda run: run.beta_kl)

beta_values = [run.beta_kl for run in runs]
beta_labels = [f"{beta:g}" for beta in beta_values]

if len(set(beta_values)) != len(beta_values):
    raise ValueError(f"Selected runs contain duplicate betaKL values: {beta_values}")


# initialize metric dictionaries
psnr_values = {beta: [] for beta in beta_values}
ssim_values = {beta: [] for beta in beta_values}
mse_values = {beta: [] for beta in beta_values}


# metrics calculation
for evaluation_source in evaluation_sources:

    evaluations = resolve_evaluations(
        runs,
        source=evaluation_source,
        checkpoint=checkpoint,
    )

    evaluations.sort(key=lambda evaluation: evaluation.beta_kl)

    # align exact same source images across betaKL values
    comparison = prepare_comparison_outputs(evaluations, gt_reference="dataset")

    print(f"\nEvaluation source: {evaluation_source.folder_name}")

    for evaluation, beta in zip(evaluations, beta_values):
        print(f"betaKL={beta:g}: {evaluation.run.name}")
        print(f"  {evaluation.folder}")

        n_images = 0

        for item in comparison.items_for(evaluation):
            gt_frames, pred_frames = comparison.load_gt_pred(item)

            for gt, pred in zip(gt_frames, pred_frames):
                n_images += 1
                
                if smooth_gt:
                    gt = gaussian_filter(gt, sigma=0.5, radius=3)

                gt[gt<-3]=-3
                gt = (gt - np.mean(gt)) / np.std(gt)
                pred = (pred - np.mean(pred)) / np.std(pred)

                data_range = np.max(gt) - np.min(gt)

                psnr_val, ssim_val, mse_val = calculate_metrics(
                    gt, pred, data_range, use_windowed,
                    window_size, patch_selection, range_invariant
                )

                psnr_values[beta].extend(psnr_val)
                ssim_values[beta].extend(ssim_val)
                mse_values[beta].extend(mse_val)

        print(f"Used #{n_images} aligned images")


# do plots
fig, axs = plt.subplots(1, 2, figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)

psnrs = [psnr_values[beta] for beta in beta_values]
ssims = [ssim_values[beta] for beta in beta_values]
mses = [mse_values[beta] for beta in beta_values]

fig = new_box_plot(psnrs, beta_labels, fig=fig, ax=axs[0],
                   ylabel="PSNR (dB)", **box_plot_parameters)

fig = new_box_plot(ssims, beta_labels, fig=fig, ax=axs[1],
                   ylabel="SSIM", ylim=(0.5, 0.95), **box_plot_parameters)

# fig = new_box_plot(mses, beta_labels, fig=fig, ax=axs[2],
#                    ylabel="MSE", **box_plot_parameters)


for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)


# show
if show_figure:
    plt.show()


# save
if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches="tight")