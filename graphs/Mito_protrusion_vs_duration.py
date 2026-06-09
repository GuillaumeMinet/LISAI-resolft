from pathlib import Path
import os,sys

import numpy as np
import matplotlib.pyplot as plt
import csv
file_path = Path(r"\\deepltestalab.dyn.scilifelab.se\e\dl_monalisa\Data\Mito_fast\MiniBranchingAnalysis\data.txt")
replace_empty_with_zeros=True

all = []
# Read data from the CSV file
with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter='\t')  # Assuming tab-separated values
    for row in reader:
        if replace_empty_with_zeros:
            y_row = [float(val) if val.strip() != '' else 0.0 for val in row] # replace empty with zeros
        else:
            y_row = [float(val) for val in row[1:] if val.strip() != '']  # Convert all y-values, skip empty
        all.append(y_row)


all = np.array(all).T  # Transpose to shape (num_profiles, num_points)

duration = [all[i] for i in range(0,3)]
lengths = [all[i] for i in range(4,7)]

labels = ["2 Hz", "1 Hz", "0.5 Hz"]
colors = ['#900400f1', "#e79b5cff", '#000000ff']

fontsize= 5.5
ticks_prms={"labelsize":fontsize, "width":0.7,"length":2}


fig=plt.figure(figsize=(1.2, 1.15))
for i in range(len(duration)):
    y = duration[i]
    x = lengths[i]
    mask=y==0
    y = y[~mask]
    x = x[~mask]/1e3
    plt.plot(y,x,'o',color=colors[i],label=labels[i],markersize=1.8)

plt.xlabel("Protrusion Duration (s)", fontsize=fontsize)
plt.ylabel("Protrusion Length (µm)", fontsize=fontsize)

plt.xticks([0,5,10,15,20])
plt.yticks([0,0.4,0.8,1.2,1.6,2.0])
plt.tick_params(axis='x',which='major',**ticks_prms)
plt.tick_params(axis='y',which='major',**ticks_prms)
plt.xlim(0,)
plt.ylim(0,)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.7)
plt.gca().spines['bottom'].set_linewidth(0.7)

# plt.show()
saving = True
# Saving
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Mito_protrusion_vs_duration.svg"
if saving:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')