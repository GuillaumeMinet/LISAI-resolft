"""

Created for 15nm image => create 4 img folders:
    - 1 x 15nm
    - 2 x 30nm : 1 with px selection, 1 by average downsizing
    - 2 x 45nm : 1 px selection and 1 avg downsizing, both re-upsampled with Nearest Neighbor

"""
import logging
import os
from pathlib import Path
from typing import Tuple, Union

from cv2 import INTER_NEAREST, resize
from scipy.ndimage import gaussian_filter
from tifffile import imread, imsave

from data.preprocess.misc.datasets_json_file import update_dataset_json
from lisai.lib.utils.get_paths import get_data_save_path

from lisai.data.utils import crop_center

logger = logging.getLogger('pipeline')

#log file header 
head = str('original_file_name new_file_name').split()
head = f'{head[0]:<30} {head[1]:<30}'

def run(dataset_name: str,
        data_format: str,
        dump_dir: Path,
        trgt_dir: Path,
        log_file_path: Path,
        combine: bool = False,
        subfolder: Path = "",
        crop_size: Union[None,int, Tuple[int, ...]] = None,
        clip_neg: bool = False,
        gauss_filter: Union[None,Tuple] = None,
        filters: Union[None,list] = None,
        ):

    assert data_format == "single", "This pipeline should be used only on single"
    
    recon_dir = dump_dir / subfolder
    if combine:
        list_folders = os.listdir(recon_dir)
        with open(log_file_path,'a') as f:
            f.write('\nCombining dataset from folders: \n')
            for folder in list_folders:
                f.write(f'\t- {str(folder)}\n')    
            f.write('\n')
           
    else:
        list_folders = [""]
        with open(log_file_path,"a") as f:
            f.write(f'\n\n{head}')
        
    count_files = 0
    for folder in list_folders:
        if combine:
            logger.info(f"folder {folder}")
            with open(log_file_path,'a') as f:
                f.write(f"\n\nFolder: {folder}\n")
                f.write(head)

        src_dir = recon_dir / folder

        data_list = sorted(os.listdir(src_dir))

        for i in range(len(data_list)):
            file_name = data_list [i]
            if file_name.split('.')[-1] not in filters:
                logger.warning(f'skipping {file_name} because not in {filters}.')
                continue 
            
            # load image and first optional preprocessing
            im = imread(src_dir / file_name)
            if crop_size is not None:
                im = crop_center(im,crop_size)
            if clip_neg:
                im[im<0] = 0
            if gauss_filter is not None and gauss_filter is not False:
                im = gaussian_filter (im,sigma = gauss_filter[0],radius = gauss_filter[1])
            

            # create the pixel selection downsampled
            dwnsamp_pxselect = im[::2,::2]
            dwnsamp_pxselect = resize(dwnsamp_pxselect, im.shape,interpolation = INTER_NEAREST )

            # create the avg downsampled
            dwnsamp_avg = im.reshape(im.shape[0]//2,2,im.shape[1]//2,2).mean(axis=(1,3))
            dwnsamp_avg = resize(dwnsamp_avg, im.shape,interpolation = INTER_NEAREST )

            # create the 45nm w/ pixel selection reupsampled to 30nm px size
            dwnsamp_inp1 = resize(im[::3,::3],dwnsamp_avg.shape,interpolation = INTER_NEAREST)
            dwnsamp_inp2 = dwnsamp_inp1.copy()
            dwnsamp_inp2[2::3, :] = 0  # Set rows 2, 5, 8, etc. to zero
            dwnsamp_inp2[:, 2::3] = 0  # Set columns 2, 5, 8, etc. to zero
            # dwnsamp_inp3 = dwnsamp_inp2.reshape(im.shape[0]//2,2,im.shape[1]//2,2).mean(axis=(1,3))

            # create the 45nm w/ pixel selection reupsampled to 30nm px size
            # dwnsamp_inp_avg = im.reshape(im.shape[0]//3,3,im.shape[1]//3,3).mean(axis=(1,3))
            # dwnsamp_inp_avg = resize(dwnsamp_inp_avg,dwnsamp_avg.shape,interpolation = INTER_CUBIC)

            # saving
            new_name = f"c{count_files:02d}"

            save_path = get_data_save_path (trgt_dir / "15nm",data_format,"recon",new_name)
            imsave(save_path, im)
            save_path = get_data_save_path (trgt_dir / "30nm_pxselect",data_format,"recon",new_name)
            imsave(save_path, dwnsamp_pxselect)
            save_path = get_data_save_path (trgt_dir / "30nm_avg",data_format,"recon",new_name)
            imsave(save_path, dwnsamp_avg)
            save_path = get_data_save_path (trgt_dir / "45nm_pxselect1",data_format,"recon",new_name)
            imsave(save_path, dwnsamp_inp1)
            save_path = get_data_save_path (trgt_dir / "45nm_pxselect2",data_format,"recon",new_name)
            imsave(save_path, dwnsamp_inp2)
            # save_path = get_data_save_path (trgt_dir / "45nm_pxselect3",data_format,"recon",new_name)
            # imsave(save_path, dwnsamp_inp3)
            

            count_files += 1
            
            logger.info(f"Processed {file_name}.")
            with open(log_file_path,"a") as f:
                f.write(f'\n {file_name:<30} {save_path.name:<30}')
        
    with open(log_file_path,"a") as f:
        f.write(f'\n\nNumber of files: {count_files}.\n')

    structure = "15nm,30nm_pxselect,30nm_avg,45nm_pxselect1,45nm_pxselect2"
    update_dataset_json(dataset_name,data_type="recon",data_format=data_format,
                        structure=structure,n_files=count_files)
