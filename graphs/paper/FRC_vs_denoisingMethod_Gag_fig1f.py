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
from graphs.utils.eval_folder import EvalSource, discover_evaluations, discover_external_evaluations
from graphs.utils.eval_outputs import prepare_comparison_outputs
from graphs.utils.paths import get_saved_graphs_dir


# data selection
dataset_name = "gag_live_tl_lowON"
model_subfolder = "HDN"
lisai_run_name = "HDN_betaKL045_00"
external_run_names = ["SN2N_Gag", "N2V_Gag"]
evaluation_source = EvalSource.dataset("gag_live_denoise_eval")
checkpoint = "best"

crop_size = 1200
smooth_gt = True

# figure parameters
conditions = ["RESOLFT 90%", "Confocal", "HDN", "SN2N", "N2V"]
colors_list = ["black", "grey", "#2e2585ff", "#48bda6ff", "#337538ff"]
show_figure = True
linewidth = 2
figure_size = (5, 5)
fontsize = 20

# saving parameters
save_figure = True
save_folder = get_saved_graphs_dir()
save_title = "Denoising_FRC_vs_Method_unsup_Gag.svg"


# discover LISAI + imported external evaluations
hdn_evaluation = discover_evaluations(dataset=dataset_name, model_subfolder=model_subfolder,
                                      run_names=[lisai_run_name], source=evaluation_source,
                                      checkpoint=checkpoint)[0]
external_evaluations = discover_external_evaluations(dataset=dataset_name, run_names=external_run_names,
                                                      source=evaluation_source)
external_by_name = {evaluation.name: evaluation for evaluation in external_evaluations}
sn2n_evaluation = external_by_name["SN2N_Gag"]
n2v_evaluation = external_by_name["N2V_Gag"]

evaluations = [hdn_evaluation, sn2n_evaluation, n2v_evaluation]
prediction_conditions = ["HDN", "SN2N", "N2V"]
comparison = prepare_comparison_outputs(evaluations)

frc_curves = {cond: [] for cond in conditions}
avg_frc_curves = {cond: [] for cond in conditions}
std_frc_curves = {cond: [] for cond in conditions}


def add_frc(arr, cond):
    if crop_size is not None: 
        arr = crop_center(arr, crop_size)
    if cond == "RESOLFT 90%" and smooth_gt: 
        arr = gaussian_filter(arr, sigma=0.5, radius=3)
    if cond in ("Confocal"): 
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


# GT and confocal belong to the evaluation dataset: load once from aligned items
count_gt = 0
count_conf = 0
img_size = None
for item in comparison.items_for(hdn_evaluation):
    gt_frames, _ = item.load_gt_pred()
    conf_frames = item.load_aux_data("conf")
    if len(gt_frames) != len(conf_frames): 
        raise ValueError("GT and confocal sample counts do not match.")
    for gt, conf in zip(gt_frames, conf_frames):
        img_size = crop_center(gt, crop_size).shape if crop_size is not None else gt.shape
        count_gt += int(add_frc(gt, "RESOLFT 90%"))
        count_conf += int(add_frc(conf, "Confocal"))
print(f"RESOLFT 90%: #{count_gt} files")
print(f"Confocal: #{count_conf} files")


# predictions from the LISAI run and imported external runs
for evaluation, cond in zip(evaluations, prediction_conditions):
    count = 0
    print(f"\n{cond}: {evaluation.name}")
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
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve,
                    alpha=0.3, color=color, linewidth=0)


ticks_prms = {"labelsize": fontsize, "width": 2, "length": 8}
ax.set_xlabel("Spatial frequency (µm$^{-1}$)", fontsize=fontsize)
ax.set_ylabel("Correlation", fontsize=fontsize)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.set_xticks([0, 5, 10, 15])
ax.tick_params(axis="x", which="major", **ticks_prms)
ax.tick_params(axis="y", which="major", **ticks_prms)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.legend()

if show_figure: 
    plt.show()
if save_figure: 
    fig.savefig(save_folder / save_title, bbox_inches="tight")
