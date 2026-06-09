import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tifffile import imread
import matplotlib.pyplot as plt

# Full path to the stack file (.tif)
folder_path = r"E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\GTqualityIssue\summary_stack_roi.tif"

# Index in the stack used as ground truth
gt_index = 6

# Optional GT smoothing
smooth_gt = False

img_stack = imread(folder_path)

n_images = int(img_stack.shape[0])


gt = img_stack[gt_index].astype(np.float64, copy=False)
if smooth_gt:
    gt = gaussian_filter (gt,sigma = 0.6,radius = 3)

gt = (gt - np.mean(gt)) / (np.std(gt))

plt.figure()
plt.imshow(gt,cmap='gray')
plt.show()

data_range = float(np.max(gt) - np.min(gt))

print(f"Stack: {folder_path}")
print(f"Shape: {img_stack.shape}")
print(f"GT index: {gt_index}")
print(f"smooth_gt: {smooth_gt}")
print("")
print("index\tPSNR(dB)\tSSIM")

for idx in range(n_images):
    if idx == gt_index:
        continue
    pred = img_stack[idx].astype(np.float64, copy=False)
    pred = (pred - np.mean(pred)) / (np.std(pred))
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim_value = structural_similarity(gt, pred, data_range=data_range)
    print(f"{idx}\t{psnr_value:.6f}\t{ssim_value:.6f}")
