""""
This pipeline has been made for the simulated datatest "simul_lines".
"""

import logging
import os
from pathlib import Path

import numpy as np
from tifffile import imread, imsave

from lisai.data.preprocess_refactor1.misc.datasets_json_file import update_dataset_json
from lisai.lib.utils.get_paths import get_data_save_path

logger = logging.getLogger('pipeline')

#log file header 
head = str('original_file_name new_file_name').split()
head = f'{head[0]:<50} {head[1]:<50}'

def run(dataset_name: str,
        data_format: str,
        dump_dir: Path,
        trgt_dir: Path,
        pxSizes: list,
        log_file_path: Path,
        file_name_template:str = "sample_and_results",
        subfolder: Path = "",
        **kwargs
        ):
    
    assert data_format == "mltpl_snr", "This pipeline should be used only on multiple snr dataset"
    recon_dir = dump_dir / subfolder 
    
    # make preprocessed data folders
    output_names = ["inp","gt_sample","gt_snr0","gt_snr1","gt_snr2"]
    for pxSize in pxSizes:
         for name in output_names:  
            folder_path = trgt_dir / f"{name}_{pxSize}nm"
            if os.path.exists(folder_path):
                logger.critical(f"{folder_path}  already exists, stopping execution."
                            "Set overwrite = True in preprocess parameters to overwrite existing folders.")
                exit()
            else:
                os.makedirs(folder_path) 

    # get list of sample folders
    sample_folders = os.listdir(recon_dir)
    sample_folders = [item for item in sample_folders if os.path.isdir(os.path.join(recon_dir, item))]
    
    # loop over all files
    n_snr_list = []
    count_files = 0
    for sample_folder in sample_folders:        
        new_name = f"c{count_files:02d}"
        for pxSize in pxSizes:
            file_name = f"{file_name_template}_{pxSize}nm.tiff"

            stack = imread(recon_dir / sample_folder / file_name)
            assert len(stack.shape) == 3 and stack.shape[0] > 1

            outputs = {
                "gt_sample" : stack[0],
                "gt_snr0" : stack[-1],
                "gt_snr1" : stack[-2],
                "gt_snr2" : stack[-3],
                "inp" : np.flip(stack[1:],axis=0)
            }

            for name,data in outputs.items():
                folder_path = trgt_dir / f"{name}_{pxSize}nm"
                save_path = get_data_save_path (folder_path, data_format,"recon",new_name)
                imsave(save_path,data)

        # update count, logger and txt file
        count_files += 1
        logger.info(f"Processed {sample_folder}.")
        with open(log_file_path,"a") as f:
            f.write(f'\n {sample_folder:<50} {new_name:<50}')

        # update n_snr list
        n_snr = outputs.get("inp").shape[0]
        if n_snr not in n_snr_list:
            n_snr_list.append(n_snr)
    
    # update dataset
    structure = ",".join([key for key in outputs])
    structure = ",".join([",".join([f"{name}_{pxSize}nm" for name in outputs]) for pxSize in pxSizes])

    update_dataset_json(dataset_name,data_type="recon",data_format=data_format,
                    structure=structure,n_files=count_files,n_snr = n_snr_list)

    
