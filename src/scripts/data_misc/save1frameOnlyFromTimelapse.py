import os

import tifffile

# Set the folder path and the timepoint to extract
folder_path = r'E:\dl_monalisa\Data\Vim_bleaching\Monalisa1\data'  # Change this to your folder
output_path = r'E:\dl_monalisa\Data\Vim_bleaching\Monalisa1\data\singletimepoint'
list_folders = ["5mW","10mW","25mW","75mW"]
T = 0  # Set the timepoint you want to extract (0-based index)

for folder in list_folders:
    
    # Create output folder
    output_folder = os.path.join(output_path, folder)
    os.makedirs(output_folder, exist_ok=True)
    list_folders = os.listdir(os.path.join(folder_path, folder))
    for filename in list_folders:
        if filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(folder_path,folder, filename)
            with tifffile.TiffFile(file_path) as tif:
                arr = tif.asarray()
                # Check if array has at least 3 dimensions (T, Y, X)
                if arr.ndim < 3:
                    print(f"Skipping {filename}: not a TYX stack.")
                    continue
                if T >= arr.shape[0]:
                    print(f"Skipping {filename}: T={T} out of bounds.")
                    continue
                single_frame = arr[T]
                # Save the single frame
                out_name = os.path.splitext(filename)[0] + f"_T{T}.tif"
                out_path = os.path.join(output_folder, out_name)
                tifffile.imwrite(out_path, single_frame)
                print(f"Saved {out_path}")