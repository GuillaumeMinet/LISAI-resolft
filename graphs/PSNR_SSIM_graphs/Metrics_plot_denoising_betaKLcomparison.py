import os, sys
sys.path.append(os.getcwd())
from lisai.graphs.utils.calculate_metrics import calculate_metrics
from lisai.graphs.utils.boxplot import box_plot

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from scipy.ndimage import gaussian_filter


full_dataset = True

# data paths


folders_path = Path(r'E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\HDN')
gt_folders_path = Path(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\preprocess\recon\gt_avg")

folder_names = [
        "HDN_single_GMMsigN2VAvgbis_KL03_noAugm",
        "HDN_single_GMMsigN2VAvgbis_KL05_noAugm",
        "HDN_single_GMMsigN2VAvgbis_KL07_noAugm",
    ]

if full_dataset:
    evaluation_folder_list = ["evaluation_best_train","evaluation_best_val","evaluation_best"]
    gt_folders_list = ["train","val","test"]
else:
    evaluation_folder_list = ["evaluation_best"]
    gt_folders_list=["test"]

betaKL_list = ["0.3", "0.5","0.7"]
colors_list = ["#2ee2f0ff","#52c2f3ff","#0a6f9cff","black"]



# metrics calculation parameters
smooth_gt = True
use_windowed = False
window_size = 600
range_invariant = False
patch_selection = 0.3


# box plot parameters
figsize = (10,5)
spaceBetweenSubplots=0.4
box_plot_parameters = {
    "colors": colors_list, # if None, each box will be all black
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
    "dataPoints_color": 'same',
    "xlabel": r"$\beta_{\mathrm{KL}}$",
    "use_mean": True,
    "labels_fontSize": 20,
    "ticks_prms": {"labelsize":20, "width":2,"length":8},
}

show_figure = False

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Denoising_metrics_vs_betaKL.svg"


# metrics calculation
psnr_values = {beta: [] for beta in betaKL_list}
ssim_values = {beta: [] for beta in betaKL_list}
mse_values = {beta: [] for beta in betaKL_list}


for idx_eval,evaluation_folder in enumerate(evaluation_folder_list):
    for idx_beta, beta in enumerate(betaKL_list):
        eval_folder = folders_path/folder_names[idx_beta]/evaluation_folder
        gt_folder = gt_folders_path / gt_folders_list[idx_eval]

        pred_list = os.listdir(eval_folder)
        gt_list = os.listdir(gt_folder)
        assert len(pred_list)//3 == len(gt_list)
        n_imgs = len(gt_list)

        for i in range (n_imgs):
            gt=imread(gt_folder/gt_list[i])
            pred=imread(eval_folder/f"img_{i}_pred.tif")

            gt[gt<-3]=-3
            gt = (gt - np.mean(gt))/ np.std(gt)
            if smooth_gt:
                gt = gaussian_filter (gt,sigma = 0.6,radius = 3)

            pred = (pred - np.mean(pred))/ np.std(pred)
            data_range = np.max(gt) - np.min(gt)
            psnr_val, ssim_val, mse_val = calculate_metrics(
                gt, pred, data_range, use_windowed,
                window_size, patch_selection, range_invariant
            )
            psnr_values[beta].extend(psnr_val)
            ssim_values[beta].extend(ssim_val)
            mse_values[beta].extend(mse_val)



# do plots

fig, axs = plt.subplots(1, 2, figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)

for _ in range(1):
    psnrs = [psnr_values[beta] for beta in betaKL_list]
    ssims = [ssim_values[beta] for beta in betaKL_list]
    mses = [mse_values[beta] for beta in betaKL_list]
    fig = box_plot(psnrs, betaKL_list, fig=fig, ax=axs[0], ylabel="PSNR (dB)",**box_plot_parameters)
    fig = box_plot(ssims, betaKL_list, fig=fig, ax=axs[1], ylabel="SSIM",**box_plot_parameters,ylim=(0.5,0.95))
    # fig = box_plot(mses, betaKL_list, fig=fig, ax=axs[2], ylabel="MSE",**box_plot_parameters)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


if show_figure:
    plt.show()      


# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')