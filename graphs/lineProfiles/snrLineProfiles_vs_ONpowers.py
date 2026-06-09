import os,sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from lisai.graphs.utils.lineProfileAnalysis import lineProfileAnalysis,readLineProfileFile
from lisai.graphs.utils.boxplot import box_plot
import math

path = r'E:\dl_monalisa\Data\Vim_bleaching\Monalisa1\data\singletimepoint'
list_powers = ["5mW","10mW","25mW","75mW"]
pxSize = 30 #nm

# line profiles plots parameters
do_plots = True # to plot individual line profile plots
show_plots = False # to show all line profiles at the end of 1 dataset processing
save_plots = False # to save all line profiles of each dataset (will be saved in the same folder as the dataset)


# finale box plot parameters
save_box_plot = True
saved_title = "SNR_Distribution_vs_405nmPower_shadedPurples.svg"
save_path = os.path.join(os.getcwd(), r"src/graphs/saved_graphs", saved_title)
save_path_bis = os.path.join(path, saved_title) # to also save in the same folder as the dataset (put to None if not needed)

box_plot_parameters = {
    "colors": ["#e6a5f1ff","#df79f1ff","#e026ecff","#ad10b8ff"], # if None, each box will be all black
    "widths": 0.3,
    "positions": [0.3, 0.8, 1.3, 1.8], # positions of the boxes on the x-axis
    "linewidth": 2,
    "dashed_whiskers": True,
    "showfliers": False,
    "showMeanAndStd": True,
    "showMeanAndStd_pos": "above",
    "showDataPoints": True,
    "dataPoints_size": 10,
    "dataPoints_alpha": 1,
    "dataPoints_color": "same",
    "labels_angle": 0,
    "xlabel": "ON-switching",
    "ylabel": "Measured SNR",
    "ylim": [1.5,14],
    "use_mean": True,
    "figsize": (8,8),
    "labels_fontSize": 25,
    "showMeanAndStd_fontsize": 18,
    "ticks_prms": {"labelsize":21, "width":2.2,"length":8},
}


snrs_all = []
for pwr in list_powers:
    print(f"Processing power: {pwr}")
    snrs_current = []
    total_profiles = 0

    # First pass to count how many plots needed
    for i in range(1, 4):
        file = f"LineProfiles{pwr}_{i}.txt"
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            _, y = readLineProfileFile(file_path, pxSize)
            total_profiles += y.shape[0]

    # Prepare shared figure for this power level
    if do_plots:
        cols = min(5, total_profiles)
        rows = math.ceil(total_profiles / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3),constrained_layout=True)
        axes = axes.flatten()
    
   # Actual processing
    profile_index = 0
    for i in range(1, 4):
        file = f"LineProfiles{pwr}_{i}.txt"
        file_path = os.path.join(path, file)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        _, snrs, _ = lineProfileAnalysis(file_path, pxSize, do_plots=do_plots,show_plots=False,#show_plots=False is for the function, to avoid showing inside the function
                                         fig=fig, axes=axes, start_index=profile_index)
        profile_index += len(snrs)
        snrs_current.extend(snrs)

    snrs_clean = [snr for snr in snrs_current if not np.isnan(snr)]
    snrs_all.append(np.array(snrs_clean))
    
    if do_plots:
        # Hide unused axes
        for j in range(profile_index, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(f"Power: {pwr}", fontsize=16)

    if save_plots:
        output_path = os.path.join(path, f"AllLineProfiles_{pwr}.svg")
        fig.savefig(output_path)
        print(f"Saved plot for {pwr} to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

avg_snrs = [np.mean(snrs) for snrs in snrs_all]
std_snrs = [np.std(snrs) for snrs in snrs_all]
print("SNR values:")
for i in range (len(list_powers)):
    print(f"{list_powers[i]}: {avg_snrs[i]} ± {std_snrs[i]}")

# Save all SNR values, averages, and stds to a txt file in column format
output_txt = os.path.join(path, "SNR_values_by_power.txt")
max_len = max(len(snrs) for snrs in snrs_all)
with open(output_txt, "w") as f:
    # Write header
    f.write("\t".join(list_powers) + "\n")
    # Write SNR values row by row
    for row in range(max_len):
        row_vals = []
        for snrs in snrs_all:
            if row < len(snrs):
                row_vals.append(f"{snrs[row]:.3f}")
            else:
                row_vals.append("")
        f.write("\t".join(row_vals) + "\n")

    # Write averages
    f.write("\n\nAverages:\n")
    avg_row = [f"{avg:.3f}" for avg in avg_snrs]
    f.write("\t".join(avg_row) + "\n")
    # Write std deviations
    f.write("\nStandard Deviations:\n")
    std_row = [f"{std:.3f}" for std in std_snrs]
    f.write("\t".join(std_row) + "\n")
print(f"Saved SNR values to {output_txt}")




# do box plot
labels=["20%","35%","60%","85%"]
fig=box_plot(snrs_all,labels,**box_plot_parameters)
# plt.show()


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2.5)
plt.gca().spines['bottom'].set_linewidth(2.5)

# save box plot
if save_box_plot:
    fig.savefig(save_path)
    try:
        if save_path_bis is not None:
            fig.savefig(save_path_bis)
    except:
        pass
