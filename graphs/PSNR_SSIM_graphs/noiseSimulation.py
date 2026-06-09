import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os

save_figure = True
save_folder = Path(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\MetricsAndNoiseExperiment")
save_title = "Vim_upsamp_FRCvsCL.svg"

# Change this path to your image file
folder_path = Path(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\inference\N2V\Vim_fixed_OnGtAvg_mse")
image_name = r"img_and_denoised_00.tiff"
image_path = folder_path / image_name
crop_size_plot = 300

def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float32)
    return (img - img_min) / (img_max - img_min)

def add_poisson_noise(img_norm: np.ndarray, peak: float) -> np.ndarray:
    noisy = np.random.poisson(img_norm * peak) / peak
    return np.clip(noisy, 0.0, 1.0)


original_gt = normalize_image(imread(image_path)[0])
clean_gt = normalize_image(imread(image_path)[1])

gt_noise_peak = 50
gt_noise_level = 1 / np.sqrt(gt_noise_peak)
noisy_gt = add_poisson_noise(clean_gt, 50)


# Zoom-in crop (centered)
h, w = clean_gt.shape
crop_size = min(crop_size_plot, h, w)
row0 = (h - crop_size) // 2
col0 = (w - crop_size) // 2
row1 = row0 + crop_size
col1 = col0 + crop_size

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
axes = axes.ravel()

axes[0].imshow(clean_gt[row0:row1, col0:col1], cmap="gray")
axes[0].set_title("Clean GT")
axes[0].axis("off")
axes[1].imshow(noisy_gt[row0:row1, col0:col1], cmap="gray")
axes[1].set_title("Noisy GT")
axes[1].axis("off")

# plt.show()

# Peak values => #counts so lower peak -> stronger Poisson noise
peak_values = [20, 10, 5]
noise_levels = [1 / np.sqrt(p) for p in peak_values]

noisy_images = [add_poisson_noise(clean_gt, peak) for peak in peak_values]
noisy_images_plot = [im[row0:row1, col0:col1] for im in noisy_images]

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
axes = axes.ravel()

for i, peak in enumerate(peak_values):
    axes[i].imshow(noisy_images_plot[i], cmap="gray")
    axes[i].set_title(f"Poisson noise (peak={peak}, crop={crop_size})")
    axes[i].axis("off")
# plt.show()


# images saving
imwrite(save_folder / "original_gt.tiff", original_gt)
imwrite(save_folder / "clean_gt.tiff", clean_gt)
imwrite(save_folder / f"noisy_gt_{gt_noise_level:02f}.tiff", noisy_gt)
for i,im in enumerate(noisy_images):
    imwrite(save_folder / f"noisy_img_{noise_levels[i]:02f}.tiff", noisy_images[i])


data_range_c = np.max(clean_gt) - np.min(clean_gt)
data_range_n = np.max(clean_gt) - np.min(clean_gt)

psnrs_clean = [psnr(clean_gt,noisy_images[i],data_range=data_range_c) for i in range(len(peak_values))]
psnrs_noisy = [psnr(noisy_gt,noisy_images[i],data_range=data_range_n) for i in range(len(peak_values))]

ssim_clean = [ssim(clean_gt,noisy_images[i],data_range=data_range_c) for i in range(len(peak_values))]
ssim_noisy = [ssim(noisy_gt,noisy_images[i],data_range=data_range_n) for i in range(len(peak_values))]

print(psnrs_clean)
print(psnrs_noisy)
print(ssim_clean)
print(ssim_noisy)





# figure parameters
figsize = (10,5)
spaceBetweenSubplots=0.5
spaceBelowSubplots=0.2
labels_fontSize=14
linewidth = 1
ticks_prms={"labelsize":labels_fontSize, "width":linewidth,"length":3*linewidth}

# do plots
fig,axs = plt.subplots(1,2,figsize=figsize)
fig.subplots_adjust(wspace=spaceBetweenSubplots)
plt.subplots_adjust(bottom=spaceBelowSubplots)


axs[0].plot(noise_levels, psnrs_clean, "ko-", label="clean GT")
axs[0].plot(noise_levels, psnrs_noisy, "ro-", label="noisy GT")
axs[0].set_title("PSNR",fontsize=labels_fontSize)

axs[1].plot(noise_levels,ssim_clean, "ko-", label="clean GT")
axs[1].plot(noise_levels,ssim_noisy, "ro-", label="noisy GT")
axs[1].set_title("SSIM",fontsize=labels_fontSize)



for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.set_xticks(noise_levels)
    ax.tick_params(axis='x',which='major',**ticks_prms)
    ax.tick_params(axis='y',which='major',**ticks_prms)
    ax.set_xlabel("Noise level",fontsize=labels_fontSize)

axs[0].set_ylabel("PSNR",fontsize=labels_fontSize)
axs[1].set_ylabel("SSIM",fontsize=labels_fontSize)


plt.show()

# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


