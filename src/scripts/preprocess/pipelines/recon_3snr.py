""""

This pipeline is for fixed dataset with different levels of snr, e.g. "high", "middle","low",
each one in different folder.

=> will create:
One inp folder "inp_mltpl_snr" where inp is a "bleaching" stack.
And 1 or 2 gt folders (depending on "gt" parameter):
- gt_snr0, and/or:
- gt_avg

NOTE: gt_snr0 will be created supposing the first folder in the list 

"""


import logging
import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter
from tifffile import imread, imsave

from data.preprocess.misc.datasets_json_file import update_dataset_json
from data.preprocess.misc.registration import pystackreg_registration
from lisai.lib.utils.get_paths import get_data_save_path

from lisai.data.utils import crop_center

logger = logging.getLogger('pipeline')

#log file header 
head = str('original_file_name new_file_name').split()
head = f'{head[0]:<50} {head[1]:<50}'

def run(dataset_name: str,
        data_format: str,
        dump_dir: Path,
        trgt_dir: Path,
        log_file_path: Path,
        combine: Path,
        subfolder: Path = "",
        list_subfolders = ["high","middle","low"],
        gt_types: Union[list, None]= None,
        registration: bool = True,
        crop_size: Union[None,int, Tuple[int, ...]] = None,
        gt_clip_neg: bool = False,
        gt_gauss_filter: Union[None,Tuple] = None,
        filters: Union[None,list] = None,
        gt_avg_exlude: int = 0,
        ):
    
    assert data_format == "mltpl_snr", "This pipeline should be used only on multiple snr dataset"
    assert not combine, "This pipeline was not written with combine option."
    recon_dir = dump_dir / subfolder 

    # make inp folder
    if os.path.exists(trgt_dir / "inp_mltpl_snr"):
            logger.critical(f"{_folder} subfolder already exists, stopping execution."
                        "Set overwrite = True in preprocess parameters to overwrite existing folders.")
            exit()
    else:
        os.makedirs(trgt_dir / "inp_mltpl_snr")
    
    # make gt folder(s)
    if gt_types is not None:
        for _type in gt_types:
            assert _type in ["snr0","avg"], "snr0 and avg are the only known types of gt"
            _folder = "gt_" + _type
            if os.path.exists(trgt_dir / _folder):
                    logger.critical(f"{_folder} subfolder already exists, stopping execution."
                                "Set overwrite = True in preprocess parameters to overwrite existing folders.")
                    exit()
            else:
                os.makedirs(trgt_dir / _folder)
    
    else:
        logger.warning("Multiple snr pipeline ongoing without creating a GT!")

    # get list of data from first subfolder of list_subfolders
    data_list = sorted(os.listdir(recon_dir / list_subfolders[0]))
    n_files = len(data_list)

    # check n_files same for all
    for subf in list_subfolders:
        assert len(os.listdir(recon_dir / subf)) == n_files, "Not the same number of files in all subfolders."

    # process each data
    for i in range(len(data_list)):
        file_name = data_list [i]
        if file_name.split('.')[-1] not in filters:
            logger.warning(f'skipping {file_name} because not in {filters}.')
            continue 

        stack = []
        for subf in list_subfolders:
            _name = sorted(os.listdir(recon_dir / subf))[i]
            stack.append(imread(recon_dir / subf / _name))
        
        stack = np.stack(stack,axis=0)
        
        if registration:
            stack = pystackreg_registration(stack) # NB: reference will always be first frame for now

        if crop_size is not None:
            stack = crop_center(stack,crop_size)
        
        # create inp NOTE: kept dict structure from mltpl_snr-pipeline, even if not really needed...
        inp_dict = dict()
        inp_dict["mltpl_snr"] = stack
                    
        # create gt(s) 
        if gt_types is not None:
            gt_dict = dict()
        if "snr0" in gt_types:
            gt_dict["snr0"] = stack[0]
        if "avg" in gt_types:
            gt_dict["avg"] = np.mean(stack[:-gt_avg_exlude if gt_avg_exlude > 0 else None],0)

        # apply gt specific preprocesses
        if gt_types is not None:
            for key in gt_dict:
                if gt_clip_neg:
                    _gt = gt_dict[key]
                    _gt[_gt<0] = 0
                    gt_dict[key] = _gt
                if gt_gauss_filter is not None and gt_gauss_filter is not False :
                    gt_dict [key] = gaussian_filter (gt_dict [key],
                                                        sigma = gt_gauss_filter[0],
                                                        radius = gt_gauss_filter[1])
        
        new_name = f"c{i:02d}"

        # saving inp(s)
        for key in inp_dict:
            save_path = get_data_save_path (trgt_dir / ("inp_" + key), data_format,
                                    "recon",new_name)
            
            imsave(save_path,inp_dict[key],imagej=True,metadata={'axes':'TYX'})

        # saving gt(s)
        if gt_types is not None:
            for key in gt_dict:
                save_path_gt = get_data_save_path (trgt_dir / ("gt_" + key), data_format,
                                        "recon",new_name)
                imsave(save_path_gt,gt_dict[key])
        

        # update logger and txt file
        logger.info(f"Processed {file_name}.")
        with open(log_file_path,"a") as f:
            f.write(f'\n {file_name:<50} {save_path.name:<50}')

    # update dataset
    n_snr_list = len(list_subfolders)
    structure = ",".join(["inp_mltpl_snr"] + ["gt_" + key for key in gt_types])

    update_dataset_json(dataset_name,data_type="recon",data_format=data_format,
                    structure=structure,n_files=n_files,n_snr = n_snr_list)

    
