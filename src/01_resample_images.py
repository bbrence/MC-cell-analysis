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
print("STEP 1: RESAMPLING IMAGES")


# ========================== 
# Load variables 
# ==========================
OUTPUT_PATH = config["output_path"]

resolutions = utils.load_variable(os.path.join(OUTPUT_PATH, "00_full_images", "org_resolutions.pkl"))


# ========================== 
# Load images 
# ==========================
mcc_list = os.listdir(f"{OUTPUT_PATH}/00_full_images/mcc")
col_list = os.listdir(f"{OUTPUT_PATH}/00_full_images/col")

mcc, col = {}, {}
for name in mcc_list:
    idx = int(name[4:6])
    mcc[idx] = sitk.ReadImage(f"{OUTPUT_PATH}/00_full_images/mcc/" + name)

for name in col_list:
    idx = int(name[4:6])
    col[idx] = sitk.ReadImage(f"{OUTPUT_PATH}/00_full_images/col/" + name)


# ==========================
# Permute axes if imaged from the side
# ==========================
IMAGED_FROM_SIDE = config["01_resample_images"]["imaged_from_side"]

if IMAGED_FROM_SIDE:
    permute_order = [2, 1, 0]
    for idx in mcc:
        org_spacing = mcc[idx].GetSpacing()

        mcc[idx] = sitk.PermuteAxes(mcc[idx], permute_order)
        col[idx] = sitk.PermuteAxes(col[idx], permute_order)

        new_spacing = tuple(org_spacing[permute_order[i]] for i in range(3))
        mcc[idx].SetSpacing(new_spacing)
        col[idx].SetSpacing(new_spacing)

    # Permute resolutions
    resolutions = [tuple(res[permute_order[i]] for i in range(3)) for res in resolutions]
    

# ==========================
# Perform resampling
# ==========================
# Determine target resolution
target_resolution = tuple(min(r[i] for r in resolutions) for i in range(3))

# Resample
mcc_resampled, col_resampled = {}, {}
for idx in tqdm(mcc, desc="Resamping"):
    mcc_resampled[idx] = utils.resample_image(mcc[idx], target_resolution)
    col_resampled[idx] = utils.resample_image(col[idx], target_resolution)

# ==========================
# Save images
# ==========================
for idx in tqdm(mcc_resampled, desc="Saving"):
    index = f"{idx:02d}"
    sitk.WriteImage(mcc_resampled[idx], f"{OUTPUT_PATH}/01_resampled_images/mcc/mcc_{index}.nrrd")
    sitk.WriteImage(col_resampled[idx], f"{OUTPUT_PATH}/01_resampled_images/col/col_{index}.nrrd")
    
    
# ========================== 
# Info
# ==========================
print(f"New resolution: {target_resolution}")
print("FINISHED")