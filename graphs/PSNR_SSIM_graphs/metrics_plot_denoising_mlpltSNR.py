import os,sys
sys.path.append(os.getcwd())
from lisai.graphs.utils.calculate_metrics import calculate_metrics
from lisai.graphs.utils.boxplot import box_plot as new_box_plot


import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from pathlib import Path





# Folder containing the image stacks
folder_path = Path(r'\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\_paper\Denoising_Technical\Different_snr\test_set')
file_names = ["c01","c14","c23"]
gt_folder = "gt"
sup_folder = "sup"
unsup_folder = "unsup"
nn_list = ["unsup","sup"]

# metrics calculation parameters
smooth_gt = True
use_windowed = True
window_size = 600
range_invariant = False
patch_selection = False


# figure parameters
colors_list = ['#009e73ff', '#d55e00ff']
box_plot_parameters = {
    "mltpl_plots":True,
    "n_plots": len(nn_list),
    "mltpl_displacements": 0.5,
    "widths": 0.3,
    "linewidth": 2,
    "dashed_whiskers": False,
    "showfliers": False,
    "showMeanAndStd": False,
    "showDataPoints": True,
    "dataPoints_size": 10,
    "dataPoints_alpha": 0.5,
    "dataPoints_color": 'same',
    "labels_angle": 0,
    "xlabel": "SNR level",
    "use_mean": True,
}

show_figure = False

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Denoising_PSNR_SSIM_MSE_vs_SNR.svg"


# load images
gts = []
for file in file_names:
    gts.append(imread(folder_path/gt_folder/f"{file}.tif"))
gts = np.stack(gts)
gts = gaussian_filter (gts,sigma = 0.6,radius = 3)
gts[gts<0]=0

sups = []
for file in file_names:
    sups.append(imread(folder_path/sup_folder/f"{file}_pred.tif")[:4])
sups = np.stack(sups)

unsups = []
for file in file_names:
    unsups.append(imread(folder_path/unsup_folder/f"{file}_pred.tif")[:4])
unsups = np.stack(unsups)

all_preds = np.stack([unsups,sups])
print(all_preds.shape)
snr_values = [1,2,3,4] 


# metrics calculation
psnr_values = {nn: {snr: [] for snr in snr_values} for nn in nn_list}
ssim_values = {nn: {snr: [] for snr in snr_values} for nn in nn_list}
mse_values = {nn: {snr: [] for snr in snr_values} for nn in nn_list}

for k in range(gts.shape[0]):
    gt = gts[k]
    preds = all_preds[:,k]

    # Normalize ground truth
    # gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
    gt =(gt - np.mean(gt))/ np.std(gt)
    data_range = np.max(gt) - np.min(gt)

    for j,nn in enumerate(nn_list):
        for i,snr in enumerate(snr_values):
            pred = preds[j,i]
            # Normalize prediction
            # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            pred = (pred - np.mean(pred)) / np.std(pred)

            psnr_val, ssim_val,mse_val = calculate_metrics(gt, pred, data_range,use_windowed,
                                                        window_size,patch_selection,range_invariant)
            psnr_values[nn][snr].extend(psnr_val)
            ssim_values[nn][snr].extend(ssim_val)
            mse_values[nn][snr].extend(mse_val)

# do plots
fig,axs = plt.subplots(1,3,figsize=(15, 5))
for plot_idx,nn in enumerate(nn_list,start=0):
    psnrs = [psnr_values[nn][snr] for snr in snr_values]
    ssims = [ssim_values[nn][snr] for snr in snr_values]
    mses = [mse_values[nn][snr] for snr in snr_values]

    fig=new_box_plot(psnrs,snr_values,fig=fig,ax=axs[0],plot_idx=plot_idx,title="PSNR",ylabel="PSNR (dB)",
                     colors=colors_list[plot_idx],**box_plot_parameters)
    fig=new_box_plot(ssims,snr_values,fig=fig,ax=axs[1],plot_idx=plot_idx,title="SSIM",ylabel="SSIM",
                     colors=colors_list[plot_idx],**box_plot_parameters)
    fig=new_box_plot(mses,snr_values,fig=fig,ax=axs[2],plot_idx=plot_idx,title="MSE",ylabel="MSE",
                     colors=colors_list[plot_idx],**box_plot_parameters)

# Custom legend
legend_names = ['HDN', 'HDN$^{sup}$']
legend_lines = [Line2D([0], [0], color=colors_list[idx], lw=1) for idx in range(len(nn_list))]
for ax in axs:
    ax.legend(legend_lines, [f'{nn}' for nn in legend_names], loc='upper right',
              fontsize=10,frameon=False,edgecolor='black')

if show_figure:
    plt.show()

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')