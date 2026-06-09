from pathlib import Path
import os,sys
sys.path.append(os.getcwd())
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from lisai.graphs.utils.lineProfileAnalysis import readLineProfileFile
import matplotlib.pyplot as plt
import numpy as np
path = Path(r"\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\_paper\Supplementary\mitoTimelapse")
file1 = "long.txt"
file2 ="short.txt"


x1,y1 = readLineProfileFile(path/file1,1,header=False,replace_empty_with_zeros=True)
x2,y2= readLineProfileFile(path/file2,1,header=False,replace_empty_with_zeros=True)

# y1=np.delete(y1,0,axis=0)
# y1=np.delete(y1,2,axis=0)

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "BleachingCurvesMitoch.svg"

save_legend = True
legend_save_title = f"{save_title.split('.')[0]}_legend.svg"


colors = ["#e6a5f1ff","#ad10b8ff"]
labels_list = ["HDN","Regular"]


mean1=[]
std1=[]
for i in range(y1.shape[1]):
    values=[y for y in y1[:,i] if y!=0]
    mean1.append(np.mean(values))
    std1.append(np.std(values))

mean2=[]
std2=[]
for i in range(y2.shape[1]):
    values=[y for y in y2[:,i] if y!=0]
    mean2.append(np.mean(values))
    std2.append(np.std(values))

mean1=np.stack(mean1)
std1=np.stack(std1)
mean2=np.stack(mean2)
std2=np.stack(std2)

fig=plt.figure(figsize=(7,5))
plt.plot(x1,mean1,color=colors[0])
plt.fill_between(range(1,len(mean1)+1),mean1-std1,mean1+std1,
                 linewidth=0,alpha=0.3,color="#e6a5f1ff")
plt.plot(x2,mean2,color=colors[1])
plt.fill_between(range(1,len(mean2)+1),mean2-std2,mean2+std2,
                 linewidth=0,alpha=0.3,color="#ad10b8ff")


plt.ylim([0,1])
axis_label_font = 25
ticks_font = 20
plt.ylabel("Norm. Intensity", fontsize=axis_label_font, fontname='Arial')
plt.xlabel("Frame number", fontsize=axis_label_font, fontname='Arial')


plt.xticks(fontsize=ticks_font, fontname='Arial', rotation=0, ha='center', va='top')
plt.xticks(fontsize=ticks_font, fontname='Arial', rotation=0, ha='center', va='top')
plt.gca().tick_params(axis='x', which='major', labelsize=ticks_font, width=2,length=8)
plt.gca().tick_params(axis='y', which='major', labelsize=ticks_font, width=2,length=8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)


plt.show()


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