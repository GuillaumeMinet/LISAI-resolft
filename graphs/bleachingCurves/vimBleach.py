import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# Path to the main folder
main_folder = r"E:\dl_monalisa\Data\Vim_bleaching\Monalisa1\data"

mean_all = []
mean_norm_all = []
err_all = []
err_norm_all = []
folder_names = ["5mW","10mW","25mW","75mW"]
colors=["#e6a5f1ff","#df79f1ff","#e026ecff","#ad10b8ff"]

labels_list = ["20%","35%","60%","85%"]

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "BleachingCurvesVimentin.svg"

save_legend = True
legend_save_title = f"{save_title}_legend.svg"


# Iterate through each folder in the main folder
for folder_name in folder_names:
    folder_path = os.path.join(main_folder, folder_name)
    intensity_mean = []
    intensity_mean_norm = []
    min_length=1000
    # Check if it's a directory
    if os.path.isdir(folder_path):

        n_files = 0
        print(f"Processing folder: {folder_name}")
        # Iterate through all .tiff files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".tiff"):
                n_files += 1
                file_path = os.path.join(folder_path, file_name)

                with tifffile.TiffFile(file_path) as tif:
                    stack = tif.asarray()

                # raw mean
                mean=np.mean(stack, axis=(1, 2))
                intensity_mean.append(mean)

                # normalized mean
                stack[stack<-5] = 0
                stack = stack - np.min(stack)
                mean_norm=np.mean(stack, axis=(1, 2))
                # mean_norm = mean_norm-np.min(mean_norm)
                mean_norm = mean_norm/mean_norm[0]
                intensity_mean_norm.append(mean_norm)

                if len(mean) < min_length:
                    min_length = len(mean)

        intensity_mean_arr = np.empty(shape=(n_files, min_length))
        intensity_mean_norm_arr = np.empty(shape=(n_files, min_length))
        for i in range(n_files):
            intensity_mean_arr[i,:] = intensity_mean[i][:min_length]
            intensity_mean_norm_arr[i,:] = intensity_mean_norm[i][:min_length]


        mean = np.mean(intensity_mean_arr,axis=0)
        mean_norm = np.mean(intensity_mean_norm_arr,axis=0)

        err = np.std(intensity_mean_arr,axis=0)
        err_norm = np.std(intensity_mean_norm_arr,axis=0)

        mean_all.append(mean)
        mean_norm_all.append(mean_norm)

        err_all.append(err)
        err_norm_all.append(err_norm)

# plt.figure()
# for i in range(len(folder_names)):
#     plt.plot(mean_all[i], label=folder_names[i],color=colors[i])
#     plt.fill_between(range(len(mean_all[i])), mean_all[i] - err_all[i], mean_all[i] + err_all[i], alpha=0.3,color=colors[i])
# plt.title(f"Bleaching Curve")
# plt.xlabel("Frame Number")
# plt.ylabel("Intensity")
# plt.legend()
# plt.show()


fig = plt.figure(figsize=(1.7,1.4))
alphas=[0.15,0.15,0.2,0.25]
for i in range(len(folder_names)):
    y = mean_norm_all[i]
    x = range(1,len(y)+1)
    plt.plot(x,y, label=folder_names[i],color=colors[i],linewidth=0.8)
    plt.fill_between(x, mean_norm_all[i] - err_norm_all[i], mean_norm_all[i] + err_norm_all[i], 
                     linewidth=0,alpha=alphas[i],color=colors[i])



axis_label_font = 5
ticks_font = 5

# plt.ylabel("Norm. Intensity", fontsize=axis_label_font, fontname='Arial')
# plt.xlabel("Frame number", fontsize=axis_label_font, fontname='Arial')

plt.ylim([0.4,1])
plt.xlim([1, 50])
plt.xticks([1, 25, 50])
plt.yticks([0.4, 0.6,0.8,1])

plt.xticks(fontsize=ticks_font, fontname='Arial', rotation=0, ha='center', va='top')
plt.xticks(fontsize=ticks_font, fontname='Arial', rotation=0, ha='center', va='top')
plt.gca().tick_params(axis='x', which='major', labelsize=ticks_font, width=0.5,length=2)
plt.gca().tick_params(axis='y', which='major', labelsize=ticks_font, width=0.5,length=2)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)


# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


# legend saving


if save_legend:

    custom_lines = [
        Line2D([0], [0], color=colors[i], lw=4, solid_capstyle='butt', label=labels_list[i])
        for i in range(len(labels_list))
    ]

    legend_fig = plt.figure(figsize=(2, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    legend = Legend(legend_ax, custom_lines, labels_list, loc='center', frameon=False, fontsize=16, handlelength=1.5, handleheight=1.0)
    legend_ax.add_artist(legend)

    legend_fig.savefig(os.path.join(save_folder, legend_save_title), bbox_inches='tight')

