import os,sys
sys.path.append(os.getcwd())
from scipy.ndimage import gaussian_filter
import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# data paths
folders_path = Path(r'E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\HDN')
gt_folders_path = Path(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\preprocess\recon\gt_avg")

folder_names = [
        "HDN_single_GMMsigN2VAvgbis_KL03_noAugm",
        "HDN_single_GMMsigN2VAvgbis_KL05_noAugm",
        "HDN_single_GMMsigN2VAvgbis_KL07_noAugm",
    ]

evaluation_folder_list = ["evaluation_best_train","evaluation_best_val","evaluation_best"]
# evaluation_folder_list = ["evaluation_best"]
gt_folders_list = ["train","val","test"]
# gt_folders_list=["test"]

betaKL_list = ["0.3", "0.5","0.7"]
colors_list = ["#2ee2f0ff","#52c2f3ff","#0a6f9cff","black"]

show_figure = False
plot_gt = False

# saving parameters
save_figure = True
save_legend = False
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Denoising_FRC_vs_betaKL_noGT.svg"
legend_save_title = f"{save_title}_legend.svg"

# intialize frc dict
frc_curves = {betaKL: [] for betaKL in betaKL_list}
avg_frc_curves = {betaKL: [] for betaKL in betaKL_list}
std_frc_curves = {betaKL: [] for betaKL in betaKL_list}

if plot_gt:
    frc_curves["gt"] = []
    avg_frc_curves["gt"] = []
    std_frc_curves["gt"] = []

# gt calculation
if plot_gt:
    for gt_folder in gt_folders_list:      
        count = 0
        eval_folder = gt_folders_path/gt_folder
        gt_files=os.listdir(eval_folder)
        n_imgs = len(gt_files)
        for i,file in enumerate(gt_files):
            print(f"GT {gt_folder} - {count+1}/{n_imgs}")
            gt = imread(eval_folder/file)
            
            gt[gt<-3]=-3
            gt = (gt-np.mean(gt))/np.std(gt)
            gt = gaussian_filter (gt,sigma = 0.6,radius = 3)
            gt[gt<0]=0
            gt = frc.util.apply_tukey(gt)
            frc_curve,_,_ = frc.one_frc(gt)
            if np.isnan(frc_curve).any():
                print(f"Skipping a patch in gt {i} due to NaN values in FRC curve.")
            else:
                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves["gt"].append(frc_curve)
            count+=1
            
    frc_curves["gt"] = np.stack(frc_curves["gt"])
    avg_frc_curves["gt"] = np.mean(frc_curves["gt"], axis=0)
    std_frc_curves["gt"] = np.std(frc_curves["gt"], axis=0)

# prediction FRC
for folder_idx,betaKL in  enumerate(betaKL_list):
    count = 0
    for evaluation_folder in evaluation_folder_list:
        eval_folder = folders_path/folder_names[folder_idx]/evaluation_folder
        n_imgs=len(os.listdir(eval_folder))//3
        for i in range(n_imgs):
            print(f"Prediction {count+1}/{n_imgs} in betaKL={betaKL}")
            pred = imread(eval_folder/f"img_{i}_pred.tif")
            pred = pred - np.min(pred)
            pred = frc.util.apply_tukey(pred)
            frc_curve,_,_ = frc.one_frc(pred)
            if np.isnan(frc_curve).any():
                print(f"Skipping a patch in image {i} due to NaN values in FRC curve.")
            else:
                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves[betaKL].append(frc_curve)
            
            count+=1
for betaKL in betaKL_list:
    frc_curves[betaKL] = np.stack(frc_curves[betaKL])
    avg_frc_curves[betaKL] = np.mean(frc_curves[betaKL], axis=0)
    std_frc_curves[betaKL] = np.std(frc_curves[betaKL], axis=0)
    


# Plot 1frc 
scale = 1/30*1e3
img_size = pred.shape
fig,ax = plt.subplots(1, 1, figsize=(5,5))
for i,betaKL in enumerate(betaKL_list):
    frc_curve = avg_frc_curves[betaKL]
    std_curve = std_frc_curves[betaKL]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label=betaKL,linewidth=1,color=colors_list[i])
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.5,color=colors_list[i],
                    linewidth=0)


# gt plot
if plot_gt:
    frc_curve = avg_frc_curves["gt"]
    std_curve = std_frc_curves["gt"]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label="gt",linewidth=0.5,color="black")
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.1,color="black",
                    linewidth=0)



labels_fontSize=20
ticks_prms={"labelsize":20, "width":2,"length":8}
ticks_positions=[0,5,10,15]
# ax.set_title("1FRC")
ax.set_xlabel("Spatial frequency (µm$^{-1}$)",fontsize=labels_fontSize)
ax.set_ylabel("Correlation",fontsize=labels_fontSize)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.legend()

ax.set_xticks(ticks_positions)
ax.tick_params(axis='x',which='major',**ticks_prms)
ax.tick_params(axis='y',which='major',**ticks_prms)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)


if show_figure:
    plt.show()  



# Saving
if save_figure:
    ax.get_legend().remove()
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


labels_list = [r"$\beta_{\mathrm{KL}}=0.3$",
               r"$\beta_{\mathrm{KL}}=0.5$",
               r"$\beta_{\mathrm{KL}}=0.7$",
               "GT"]
if save_legend:
    custom_lines = [
        Line2D([0], [0], color=colors_list[i], lw=4, solid_capstyle='butt', label=labels_list[i])
        for i in range(len(labels_list))
    ]

    legend_fig = plt.figure(figsize=(2, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    legend = Legend(legend_ax, custom_lines, labels_list, loc='center', frameon=False, fontsize=16, handlelength=1.5, handleheight=1.0)
    legend_ax.add_artist(legend)

    legend_fig.savefig(os.path.join(save_folder, legend_save_title), bbox_inches='tight')