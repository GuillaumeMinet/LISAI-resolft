import logging

logger = logging.getLogger("change_name")

def change_name(name,transf,filters,suffixes):

    if transf == "folder_to_file":
        # meaning raw to recon
        new_name = name + suffixes["recon"]
    
    elif transf == "file_to_folder":
        # meaning recon to raw
        new_name = name.split('.')[0]

    elif transf == "file_to_file":
        # could be both case, need to differentiate
        if name.split('.')[-1] in filters["recon"]:
            # recon to raw
            new_name = name.split('.')[0] + suffixes["raw"]
        elif name.split('.')[-1] in filters["raw"]:
            # raw to recon
            new_name = name.split('.')[0] + suffixes["recon"]
        else:
            name = {name.split('.')[-1]}
            logger.critical(f'Changing name issue: {name}'
                            f' is not in any filters. Returning None for now, '
                            f'but this might create issues later on.')
            return None
    else:
        raise ValueError(f"transf {transf} for changing name is not known, should be"
                        f" one of those 3: 'file_to_folder','file_to_file, 'folder_to_file'")
        
    return new_name
