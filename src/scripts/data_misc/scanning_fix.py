import copy

import numpy as np
import tifffile as tiff

# Path to the image
image_path = r"E:\dl_monalisa\Data\Mito_fast\20250321\mito_fast\c6.tiff"

# Open the image
img_stack = tiff.imread(image_path)

# Apply the pixel removal operation to the entire 3D stack
arr_stack = copy.deepcopy(img_stack)
arr_stack = np.delete(arr_stack, np.arange(9, arr_stack.shape[2], 10), axis=2)  # Remove every 10th pixel along x-axis
arr_stack = np.delete(arr_stack, np.arange(9, arr_stack.shape[1], 10), axis=1)  # Remove every 10th pixel along y-axis

tiff.imsave(r"E:\dl_monalisa\Data\Mito_fast\20250321\mito_fast\c6_fixed.tiff", arr_stack)

# # Display the first frame of the modified stack for visualization (optional)
# plt.subplot(1, 2, 1)
# plt.imshow(arr_stack[0], cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img_stack[0], cmap='gray')
# plt.axis('off')  # Optional: Turn off axis labels
# plt.show()

