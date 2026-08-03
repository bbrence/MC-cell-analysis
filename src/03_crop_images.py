# ========================== 
# Imports 
# ==========================
import os
from tqdm import tqdm
import SimpleITK as sitk

import utils


# ========================== 
# Load config 
# ==========================
config = utils.import_config()


# ========================== 
# Info
# ==========================
print("STEP 3: CROPPING IMAGES")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

mcc_list = os.listdir(f"{OUTPUT_PATH}/02_registered_images/mcc")
col_list = os.listdir(f"{OUTPUT_PATH}/02_registered_images/col")

mcc, col = {}, {}
for name in mcc_list:
    idx = int(name[4:6])
    mcc[idx] = sitk.ReadImage(f"{OUTPUT_PATH}/02_registered_images/mcc/" + name)

for name in col_list:
    idx = int(name[4:6])
    col[idx] = sitk.ReadImage(f"{OUTPUT_PATH}/02_registered_images/col/" + name)
    

# ========================== 
# Crop images 
# ==========================
MIN_IDX = config["03_crop_images"]["min_idx"]
MAX_IDX = config["03_crop_images"]["max_idx"]

size = []
for min_i, max_i in zip(MIN_IDX, MAX_IDX):
    size.append(max_i - min_i + 1)  # +1 to include max_i
   
# Cropping
mcc_cropped, col_cropped = {}, {}
for idx in tqdm(mcc, desc="Cropping"):
    mcc_cropped[idx] = sitk.RegionOfInterest(mcc[idx], size, MIN_IDX)
    col_cropped[idx] = sitk.RegionOfInterest(col[idx], size, MIN_IDX)
    

# ========================== 
# Save cropped images 
# ==========================   
for idx in tqdm(mcc, desc="Saving"):
    index = f"{idx:02d}"
    sitk.WriteImage(mcc_cropped[idx], f"{OUTPUT_PATH}/03_cropped_images/mcc/mcc_{index}.nrrd")
    sitk.WriteImage(col_cropped[idx], f"{OUTPUT_PATH}/03_cropped_images/col/col_{index}.nrrd")
    

# ========================== 
# Info
# ==========================
print(f"New image size: {size}")
print("FINISHED")