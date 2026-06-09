import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os



# File path
file_path = r"\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\_paper\Supplementary\SNR_onSwitching_SNRs\monalisa1_onRamp.txt"

# saving parameters
save_figure = True
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "ONswitchingCurvesVimentin.svg"

# Read the data
x = []
y = []
with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split('\t')
        x.append(float(columns[1])/1000)
        y.append(float(columns[2].replace(',', '.')))

# Plot the data ()
fig = plt.figure(figsize=(9, 6))
plt.plot(x, y, marker='o', linestyle='--', color='black', label='Activation Curve')


axis_label_font = 25
ticks_font = 20
plt.ylabel("Norm. Intensity", fontsize=axis_label_font, fontname='Arial')
plt.xlabel("Power density (kW/cm2)", fontsize=axis_label_font, fontname='Arial')


plt.xticks(fontsize=ticks_font, fontname='Arial', rotation=0, ha='center', va='top')
plt.xticks(fontsize=ticks_font, fontname='Arial', rotation=0, ha='center', va='top')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().tick_params(axis='x', which='major', labelsize=ticks_font, width=2,length=8)
plt.gca().tick_params(axis='y', which='major', labelsize=ticks_font, width=2,length=8)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)




# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')