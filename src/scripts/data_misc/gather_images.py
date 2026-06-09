import os
from pathlib import Path

import numpy as np
import tifffile as tiff


def gather_image_stack(models_folder, folder_list, save_folder,normalization = False):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if folder_list == "all":
        folder_list = [f for f in os.listdir(models_folder) if os.path.isdir(os.path.join(models_folder, f))]

    folder_txt_path = os.path.join(save_folder, 'models_list.txt')
    with open(folder_txt_path, 'w') as f:
        for folder in folder_list:
            f.write(f'{folder}\n')

    # Get the list of image files from the first folder (ground truth is the same for all models)
    first_folder_path = os.path.join(models_folder, folder_list[0], 'evaluation_last')
    img_files = [f for f in os.listdir(first_folder_path) if f.endswith('_inp.tif')]

    for img_file in img_files:
        img_index = img_file.split('_')[1]
        gt_img_path = os.path.join(first_folder_path, f'img_{img_index}_gt.tif')

        # Load ground truth image (gt), only need to do it once
        gt_img = tiff.imread(gt_img_path)
        if normalization:
            gt_img = (gt_img - np.mean(gt_img)) / np.std(gt_img)
        
        # Create list to hold the stacked images
        image_stack = [gt_img]

        # For each folder, load pred images and add to stack
        for folder_name in folder_list:
            pred_img_path = os.path.join(models_folder, folder_name, 'evaluation_last', f'img_{img_index}_pred.tif')
            pred_img = tiff.imread(pred_img_path)
            if normalization:
                pred_img = (pred_img - np.mean(pred_img)) / np.std(pred_img)
            image_stack.append(pred_img)
        
        # Stack images and save
        image_stack = np.stack(image_stack, axis=0)
        save_path = os.path.join(save_folder, f'results_summary_img{img_index}.tif')
        tiff.imwrite(save_path, image_stack,imagej=True,metadata={"axes":"TYX"})

if __name__ == "__main__":
    
    models_folder = r"E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_selected\unpaired"
    folder_list = "all"
    save_folder = Path(models_folder)
    gather_image_stack(models_folder, folder_list, save_folder,normalization = True)
