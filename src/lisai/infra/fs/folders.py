# src/lisai/fs/folders.py
import logging
import shutil
from pathlib import Path

from lisai.infra.paths import Paths as LisaiPaths

from .output_folders import prepare_output_folder
from .run_naming import (
    allocate_run_dir_name,
    format_run_dir_name,
    get_unique_exp_name,
    run_dir_index_width,
)

logger = logging.getLogger("lisai.fs")


def ensure_folder(path: Path, mode: str = "exist_ok") -> Path:
    """
    Creates a folder with specific safety behaviors.

    Args:
        path (Path): The directory path to create.
        mode (str): 
            - "strict": The folder MUST NOT exist. Raises FileExistsError if it does.
            - "exist_ok": It is okay if the folder exists (standard mkdir -p).
            - "overwrite": If folder exists, DELETE IT and recreate + logs a warning.

    Returns:
        Path: The confirmed path object.
    
    Raises:
        FileExistsError: if mode="strict" and folder exists.
        NotADirectoryError: if folder exists, mode == "exist_ok" but it's not a directory
    """
    path = Path(path)

    if path.exists():
        if mode == "strict":
            raise FileExistsError(f"Folder already exists at '{path}' but mode is STRICT.")

        elif mode == "overwrite":
            logger.warning(f"\nSAVING: Overwriting existing folder: {path}\n")
            shutil.rmtree(path)

        elif mode == "exist_ok":
            if not path.is_dir():
                raise NotADirectoryError(f"Path exists but is not a directory: {path}")
            return path
        else:
            raise ValueError(f"Unknown mode: {mode}")

    path.mkdir(parents=True, exist_ok=False)
    return path


def create_run_dir(paths: LisaiPaths, ds_name: str, exp_name: str, subfolder: str = "", overwrite: bool = False):
    """
    Creates unique run folder using run_name + index naming (`run_name_00`).
    Args:
        paths (Paths object): runtime paths
        ds_name (str): dataset name
        exp_name (str): semantic run name
        subfolder (str): dataset subfolder
        overwrite (bool): to overwrite previous savings with same exp_name
    Returns:
        path: final full saving path
        exp_name: final run directory name
    """
    intended_run_dir = paths.run_dir(dataset_name=ds_name,
                                 models_subfolder=subfolder,
                                 exp_name=exp_name)
    runs_root = intended_run_dir.parent
    ensure_folder(runs_root,mode="exist_ok")

    if overwrite:
        final_name = format_run_dir_name(exp_name, 0, width=run_dir_index_width())
        if_exists_policy = "overwrite"
    else:
        final_name, _ = allocate_run_dir_name(runs_root, exp_name, width=run_dir_index_width())
        if_exists_policy = "error"

    final_run_dir = runs_root / final_name
    resolution = prepare_output_folder(final_run_dir, if_exists_policy=if_exists_policy)
    return resolution.path, final_name


def create_tb_folder(tb_folder: Path, exp_name: str, exist_ok: bool = True):
    """
    Creates tensorboard folder. If exist_ok is False, we keep exp_name as given,
    otherwise we auto-increment it to have unique folder name. 
    Args:
        tb_folder (Path): saving directory
        exp_name (str): raw experiment name
    Returns:
        path: final full saving path (tb_folder / exp_name)
        exp_name: final exp_name
    """
    tb_folder = Path(tb_folder)

    if not exist_ok:
        exp_name = get_unique_exp_name(tb_folder, exp_name)

    path = tb_folder / exp_name
    resolution = prepare_output_folder(
        path,
        if_exists_policy="reuse" if exist_ok else "error",
    )
    return resolution.path, exp_name
