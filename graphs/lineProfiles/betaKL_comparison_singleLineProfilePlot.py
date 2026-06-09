import sys,os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from lisai.graphs.utils.lineProfileAnalysis import readLineProfileFile
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
path = r"\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\_paper\Supplementary\betaKL_comparison\lineProfile_2.txt"
pxSize=0.03# um
colors_list = ["#2ee2f0ff", "#52c2f3ff","#0a6f9cff", "black"]

labels_list = [r"$\beta_{\mathrm{KL}}=0.7$",
               r"$\beta_{\mathrm{KL}}=0.5$",
               r"$\beta_{\mathrm{KL}}=0.3$",
               "GT"]

# saving
save_figure = True
save_legend = False
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Denoising_SingleLineProfile_diffBeta_2.svg"
legend_save_title = f"{save_title}_legend.svg"


# plot data
x,y = readLineProfileFile(path, pxSize,header=False)
x_start = 8
x_stop = - 6
x = x[x_start:x_stop] - x[x_start]

fig = plt.figure(figsize=(3,3))

for col in range(y.shape[0]):
    y_values = y[col,x_start:x_stop]
    y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    plt.plot(x,y_values,color=colors_list[col],label=labels_list[col])

plt.xlabel(r"$x\,(\mu\mathrm{m})$", fontsize=16, fontname='Arial')
plt.ylabel("Norm. Intensity", fontsize=16, fontname='Arial')

plt.xticks(fontsize=12, fontname='Arial', rotation=0, ha='center', va='top')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().tick_params(axis='x', which='major', labelsize=15, width=2)
plt.gca().tick_params(axis='y', which='major', labelsize=15, width=2)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)


# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')

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