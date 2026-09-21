from pathlib import Path

import numpy as np
from tifffile import imread, imsave

from lisai.data.utils import get_saving_shape

# Input folder containing .tif files
input_folder = Path(r"E:\dl_monalisa\Data\Mito_fast\20250326\Predict_Upsampling_CL5_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005")

# Output folder to save modified images
output_folder = input_folder / "withBlackFrameFirst"
output_folder.mkdir(parents=True, exist_ok=True)

# Iterate over all .tif files in the input folder
for file_path in input_folder.glob("*.tif"):
    # Read the .tif file
    image = imread(file_path)

    # Print dimensions of the TIF file for verification
    print(f"File: {file_path.name} | Dimensions: {image.shape}")
    
    # Check for 4D images and infer dimension order
    if len(image.shape) == 4:
        if image.shape[0] < image.shape[1]:  # Guess that Z is smaller and Z,T order
            order = 'Z,T,X,Y'
            z_axis = 0
            t_axis = 1
        else:  # T,Z order
            order = 'T,Z,X,Y'
            z_axis = 1
            t_axis = 0

        print(f"Inferred dimension order: {order} (Z-axis: {z_axis}, T-axis: {t_axis})")
        
        # Add an extra dummy slice before the first Z frame for all T
        if order == 'Z,T,X,Y':
            dummy_slice = np.zeros((1, image.shape[1], image.shape[2], image.shape[3]), dtype=image.dtype)
            modified_image = np.concatenate([dummy_slice, image], axis=0)
        elif order == 'T,Z,X,Y':
            dummy_slice = np.zeros((image.shape[0], 1, image.shape[2], image.shape[3]), dtype=image.dtype)
            modified_image = np.concatenate([dummy_slice, image], axis=1)
        
        # Save the modified image to the output folder
        output_path = output_folder / file_path.name
        shape = get_saving_shape(modified_image)
        imsave(output_path, modified_image,imagej=True,
                metadata={"axes": shape})
        print(f"Modified image saved to: {output_path} | New Dimensions: {modified_image.shape}")
    else:
        print(f"Skipping: {file_path.name} (Not a 4D image)")
