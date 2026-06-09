import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lisai.data.utils import crop_center
import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
from scipy.ndimage import gaussian_filter

folder_path= r"E:\lisai\inference\Gag\2026-03-06_conf_low_high"

folders = [
    "high",
    "conf",
    "Predict_HDN_HDN_betaKL04_00",
    "SN2N",
    "N2V",
]

conditions = [
    "RESOLFT 90%", 
    "Confocal", 
    "HDN",
    "SN2N",
    "N2V",
    ]

crop_size = 1200
smooth_gt = True

show_figure = True
# saving parameters

save_figure = True
save_legend = False
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Gag_upsamp_highSNR_FRC_withSN2NandN2V.svg"
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
        if cond == "HDN":
            if "pred" not in img_name:
                continue
        arr = imread(path / img_name)
        if cond == "N2V":
            arr = arr[-1]
        if cond == "SN2N":
            arr = arr[0,0]
        if crop_size is not None:
            arr = crop_center(arr,crop_size)
            print(arr.shape)
        if cond == "RESOLFT 90%":
            if smooth_gt:
                arr = gaussian_filter (arr,sigma = 0.5,radius = 3)
        if cond == "RESOLFT 90%" or cond == "Confocal":
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

colors_list = ['black','grey',"#2e2585ff",'#48bda6ff','#337538ff']#,'forestgreen']


# Plot 1frc 
scale = 1/30*1e3
img_size = arr.shape
fig,ax = plt.subplots(1, 1, figsize=(5, 5))
for cond in conditions:
    frc_curve = avg_frc_curves[cond]
    std_curve = std_frc_curves[cond]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label=cond,linewidth=2,color=colors_list[conditions.index(cond)])
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.3,color=colors_list[conditions.index(cond)],
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


if save_figure:
    fig.savefig(save_folder / save_title, bbox_inches='tight')
