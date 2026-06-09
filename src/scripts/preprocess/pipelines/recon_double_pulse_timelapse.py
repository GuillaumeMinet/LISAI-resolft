import logging
import os
from pathlib import Path
from typing import Tuple, Union

from tifffile import imread, imsave

from data.preprocess.misc.datasets_json_file import update_dataset_json
from lisai.lib.utils.get_paths import get_data_save_path

from lisai.data.utils import crop_center

logger = logging.getLogger('pipeline')

#log file header 
head = str('original_file_name new_file_name (same for inp and gt)').split()
head = f'{head[0]:<30} {head[1]:<30}'




def run(dataset_name: str,
        data_format: str,
        dump_dir: Path,
        trgt_dir: Path,
        log_file_path: Path,
        combine: bool = False,
        subfolder_inp: Path = "",
        subfolder_gt: Path = "",
        crop_size: Union[None,int, Tuple[int, ...]] = None,
        filters: Union[None,list] = None):
    

    assert data_format == "timelapse","This pipeline should be used only on timelapse dataset"
    assert combine is False, "Combine option not coded yet in this pipeline"

    src_dir_inp = dump_dir / subfolder_inp
    src_dir_gt = dump_dir / subfolder_gt

    # create inp and gt folders
    if os.path.exists(trgt_dir / "inp"):
        logger.critical("'inp' folder already exists, stopping execution."
            "Set overwrite = True in preprocess parameters to overwrite existing folders.")
        exit()
    if os.path.exists(trgt_dir / "gt"):
        logger.critical("'gt' folder already exists, stopping execution."
            "Set overwrite = True in preprocess parameters to overwrite existing folders.")
        exit()

    os.makedirs(trgt_dir / "inp")
    os.makedirs(trgt_dir / "gt")
   
    # get list of inp
    data_list_inp = sorted(os.listdir(src_dir_inp))
    for i in range(len(data_list_inp)):
        file_name = data_list_inp [i]
        if file_name.split('.')[-1] not in filters:
            logger.warning(f'Removing {file_name} from inp list because type not in {filters}.')
            data_list_inp.remove(file_name) 
    # get list of gt
    data_list_gt = sorted(os.listdir(src_dir_gt))
    for i in range(len(data_list_gt)):
        file_name = data_list_gt [i]
        if file_name.split('.')[-1] not in filters:
            logger.warning(f'Removing {file_name} from gt list because type not in {filters}.')
            data_list_gt.remove(file_name) 
    # check number of files in gt and inp is the same
    assert len(data_list_gt) == len(data_list_inp), "not the same number of files in gt and inp folders!"

    # log file header
    with open(log_file_path,"a") as f:
        f.write(f'\n\n{head}')


    # process each file and save in preprocess folder
    count_files = 0
    count_frames = 0
    for i in range(len(data_list_inp)):
        file_name = data_list_inp [i]

        # read
        stack_inp = imread(src_dir_inp / file_name)
        stack_gt = imread(src_dir_gt / file_name)
        
        # checks
        assert len(stack_inp.shape) == 3 and stack_inp.shape[0] > 1, "Data should be timelapses"
        assert stack_inp.shape[0] == stack_gt.shape[0], "GT and INP should have the same number of frames"
        
        # crop
        if crop_size is not None:
            stack_inp = crop_center(stack_inp,crop_size)
            stack_gt = crop_center(stack_gt,crop_size)
        
        # saving
        t = stack_inp.shape[0] #number of time points
        new_name = f"c{count_files:02d}"
        save_path_inp = get_data_save_path (trgt_dir / "inp",data_format,"recon",new_name,number_of_timepoints=t)
        save_path_gt = get_data_save_path (trgt_dir / "gt",data_format,"recon",new_name,number_of_timepoints=t)
        
        imsave(save_path_inp, stack_inp)
        imsave(save_path_gt, stack_gt)

        # update count, logger and txt file
        count_files += 1
        count_frames += t
        logger.info(f"Processed {file_name}.")
        with open(log_file_path,"a") as f:
            f.write(f'\n {file_name:<30} {save_path_inp.name:<30}')
    
    with open(log_file_path,"a") as f:
        f.write(f'\n\nNumber of files: {count_files}.\n'
                f'Total number of frames: {count_frames}.')
    

    structure = "inp,gt"

    update_dataset_json(dataset_name,data_type="recon",data_format=data_format,
                        structure=structure,n_files=count_files, n_frames=count_frames)
