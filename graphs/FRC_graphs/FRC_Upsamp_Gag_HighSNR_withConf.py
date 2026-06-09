import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphs.utils.eval_folder import get_eval_folder, list_images
from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot as new_box_plot
from lisai.data.utils import crop_center, extract_patches

import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
import random

from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from scipy.ndimage import gaussian_filter

folder_path= r"E:\lisai\inference\Gag\2026-03-07_high_snr"

folders = [
    "resolft",
    "conf",
    "Predict_Upsamp_CL1_Upsamp2_biggerNet_02_01",
    "Predict_Upsamp_CL1_Upsamp05_biggerNet_01_01"
]

conditions = [
    "GT", 
    "Confocal", 
    "S=0.25",
    "S=0.5"
    ]

crop_size = 1218
smooth_gt = True

show_figure = True
# saving parameters
save_figure = True
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Gag_upsamp_highSNR_FRC.svg"
legend_save_title = f"{save_title.split('.')[0]}_legend.svg"


frc_curves = {cond: [] for cond in conditions}
avg_frc_curves = {cond: [] for cond in conditions}
std_frc_curves = {cond: [] for cond in conditions}

for idx,f in enumerate(folders):
    path = Path(folder_path) / f
    imgs_list = os.listdir(path)
    count=0
    cond = conditions[idx]
    for img_name in imgs_list:
        arr = imread(path / img_name)
        if crop_size is not None:
            arr = crop_center(arr,crop_size)

        if cond == "GT":
            if smooth_gt:
                arr = gaussian_filter(arr,sigma = 0.3,radius = 3)
        
        if cond == "GT" or cond == "Confocal":
            arr[arr<0]=0
            arr = (arr - np.mean(arr)) / np.std(arr)
            arr = arr - np.min(arr)
        else:
            arr = (arr - np.mean(arr)) / np.std(arr)
            arr = arr - np.min(arr)
            
        arr = frc.util.apply_tukey(arr)
        frc_curve = frc.one_frc(arr)
        if np.isnan(frc_curve).any():
            print(f"Skipping image due to NaN values in FRC curve.")
        else:
            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
            frc_curves[cond].append(frc_curve)
            count+=1
    print(f"Folder {f}: #{count} files")


for cond in conditions:
    frc_curves[cond] = np.stack(frc_curves[cond])
    avg_frc_curves[cond] = np.mean(frc_curves[cond], axis=0)
    std_frc_curves[cond] = np.std(frc_curves[cond], axis=0)

colors_list = ['black','grey','#a70048ff','#0000ffff']#,'forestgreen']


# Plot 1frc 
scale = 1/30*1e3
img_size = arr.shape
fig,ax = plt.subplots(1, 1, figsize=(1, 1))
for cond in conditions:
    frc_curve = avg_frc_curves[cond]
    std_curve = std_frc_curves[cond]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label=cond,linewidth=0.4,color=colors_list[conditions.index(cond)])
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.3,color=colors_list[conditions.index(cond)],
                    linewidth=0)
    

fontsize=5.5
xticks_positions = [0,5,10,15]
yticks_positions = [0,0.5,1]
ticks_prms={"labelsize":fontsize, "width":0.7,"length":2}

# ax.set_title("1FRC")
ax.set_xlabel("Spatial frequency (µm$^{-1}$)",fontsize=fontsize)
ax.set_ylabel("Correlation",fontsize=fontsize)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
# ax.legend()

ax.set_xticks([0,5,10,15])
ax.set_yticks([0,0.5,1])
ax.tick_params(axis='x',which='major',**ticks_prms)
ax.tick_params(axis='y',which='major',**ticks_prms)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.7)
plt.gca().spines['bottom'].set_linewidth(0.7)


if show_figure:
    plt.show()  
# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')