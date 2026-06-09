import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import matplotlib.pyplot as plt
from graphs.utils.lineProfileAnalysis import lineProfileAnalysis,readLineProfileFile
from graphs.utils.boxplot import box_plot
import math

path = r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\Evaluation\img1_testsplit_lineProfilesUpsamp"
list_cond = ["GT","025","05","075"]
pxSize = 28 # nm
fit_method = "lorentzian"

# line profiles plots parameters
do_plots = True # put false if you don't need to show or save all the line profiles plots
show_plots = False # to show all line profiles at the end of each dataset processing
save_plots = True # to save all line profiles of each dataset (will be saved in the same folder as the dataset)

show_final_boxplot = True

# finale box plot parameters
save_box_plot = True
saved_title = "LineProfiles_upsampPred.svg"
save_folder = PROJECT_ROOT / "graphs" / "saved_graphs"
save_title = "Vim_fixed_lineProfiles_Upsamp.svg"

save_path_bis = None # os.path.join(path, saved_title) # to also save in the same folder as the dataset (put to None if not needed)

box_plot_parameters = {
    "colors": ["darkred","mediumblue","forestgreen","black"], 
    "widths": 0.2,
    "positions": [0.3,0.7,1.1,1.4], # positions of the boxes on the x-axis
    "linewidth": 2,
    "dashed_whiskers": True,
    "showfliers": False,
    "showMeanAndStd": False,
    "showMeanAndStd_pos": "above",
    "showDataPoints": True,
    "dataPoints_size": 20,
    "dataPoints_alpha": 1,
    "dataPoints_color": 'same',
    "labels_angle": 0,
    "xlabel": "Sampling",
    "ylabel": "FWHM",
    "ylim": [40,130],
    "use_mean": True,
    "reverse_dataset_order":False,
    "custom_order": [1,2,3,0],
    "figsize": (6.5,8.5),
    "labels_fontSize": 25,
    "showMeanAndStd_fontsize": 18,
    "ticks_prms": {"labelsize":20, "width":2,"length":8},
}


fwhms_all = []
for cond in list_cond:
    print(f"Processing condition: {cond}")
    fwhms_current = []
    total_profiles = 0

    # First pass to count how many plots needed
    for i in range(1, 2):
        file = f"{cond}.txt"
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            _, y = readLineProfileFile(file_path, pxSize)
            total_profiles += y.shape[0]
        else:
            print(f"File {file_path} does not exist. Skipping.")
            continue

    # Prepare shared figure for this power level
    if do_plots:
        cols = min(5, total_profiles)
        rows = math.ceil(total_profiles / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3),constrained_layout=True)
        axes = axes.flatten()
    
   # Actual processing
    profile_index = 0
    for i in range(1, 2):
        file = f"{cond}.txt"
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        fwhms, _, _ = lineProfileAnalysis(file_path, pxSize, do_plots=do_plots,show_plots=False,#show_plots=False is for the function, to avoid showing inside the function
                                         fig=fig, axes=axes, start_index=profile_index,calculate_snr=False,
                                         fit_method=fit_method)
        profile_index += len(fwhms)
        fwhms_current.extend(fwhms)

    fwhms_clean = [fwhm for fwhm in fwhms_current if not np.isnan(fwhm)]
    fwhms_all.append(np.array(fwhms_clean))
    
    if do_plots:
        # Hide unused axes
        for j in range(profile_index, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(f"{cond}", fontsize=16)

    if save_plots:
        output_path = os.path.join(path, f"AllLineProfiles_{cond}.svg")
        fig.savefig(output_path)
        print(f"Saved plot for {cond} to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

avg_fwhms = [np.mean(fwhms) for fwhms in fwhms_all]
std_fwhms = [np.std(fwhms) for fwhms in fwhms_all]
print("FWHM values:")
for i in range (len(list_cond)):
    print(f"{list_cond[i]}: {avg_fwhms[i]} ± {std_fwhms[i]}")

# Save all FWHM values, averages, and stds to a txt file in column format
output_txt = os.path.join(path, "SNR_values_by_snrLvl.txt")
max_len = max(len(fwhms) for fwhms in fwhms_all)
with open(output_txt, "w") as f:
    # Write header
    f.write("\t".join(list_cond) + "\n")
    # Write SNR values row by row
    for row in range(max_len):
        row_vals = []
        for fwhm in fwhms_all:
            if row < len(fwhm):
                row_vals.append(f"{fwhm[row]:.3f}")
            else:
                row_vals.append("")
        f.write("\t".join(row_vals) + "\n")

    # Write averages
    f.write("\n\nAverages:\n")
    avg_row = [f"{avg:.3f}" for avg in avg_fwhms]
    f.write("\t".join(avg_row) + "\n")
    # Write std deviations
    f.write("\nStandard Deviations:\n")
    std_row = [f"{std:.3f}" for std in avg_fwhms]
    f.write("\t".join(std_row) + "\n")
print(f"Saved FWHM values to {output_txt}")


# do box plot
fig=box_plot(fwhms_all,list_cond,**box_plot_parameters)
# plt.show()

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

if show_final_boxplot:
    plt.show()

# save box plot
if save_plots:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')
    try:
        if save_path_bis is not None:
            fig.savefig(save_path_bis)
    except:
        pass