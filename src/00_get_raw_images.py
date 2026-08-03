# ========================== 
# Imports 
# ==========================
import os
import SimpleITK as sitk

import utils


# ========================== 
# Load config 
# ==========================
config = utils.import_config()


# ==========================
# Get input image list, get output path
# ==========================
INPUT_PATH = config["00_get_raw_images"]["input_path"]
input_files = sorted(os.listdir(INPUT_PATH))

OUTPUT_PATH = config["output_path"]

# ========================== 
# Info
# ==========================
print("STEP 0: GETING RAW IMAGES FROM MICROSCOPY IMAGES")
print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")


# ==========================
# Create folder structure 
# ==========================
# Single images save folder
os.makedirs(os.path.join(OUTPUT_PATH, "00_full_images", "mcc"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "00_full_images", "col"), exist_ok=True)

# Resampled images save folder
os.makedirs(os.path.join(OUTPUT_PATH, "01_resampled_images", "mcc"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "01_resampled_images", "col"), exist_ok=True)

# Registered images save folder
os.makedirs(os.path.join(OUTPUT_PATH, "02_registered_images", "mcc"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "02_registered_images", "col"), exist_ok=True)

# Cropped images save folder
os.makedirs(os.path.join(OUTPUT_PATH, "03_cropped_images", "mcc"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "03_cropped_images", "col"), exist_ok=True)

# Second round registered images save folder
os.makedirs(os.path.join(OUTPUT_PATH, "04_registered_images_2", "mcc"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "04_registered_images_2", "col"), exist_ok=True)

# Thresholded images, ONLY MCC
os.makedirs(os.path.join(OUTPUT_PATH, "05_thresholded_images", "mcc"), exist_ok=True)

# Photoreceptor centers, ONLY COLS
os.makedirs(os.path.join(OUTPUT_PATH, "06_corrected_r7_centers", "col"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "06_corrected_r7_centers", "centers"), exist_ok=True)

# Adjusted angles
os.makedirs(os.path.join(OUTPUT_PATH, "07_rotated_binary_images", "mcc"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "07_rotated_binary_images", "mcc_2d_proj"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "07_rotated_binary_images", "col"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "07_rotated_binary_images", "centers"), exist_ok=True)

# Voronoi regions
os.makedirs(os.path.join(OUTPUT_PATH, "08_voronoi_regions", "voronoi"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "08_voronoi_regions", "voronoi_2d"), exist_ok=True)

# Voronoi regions
os.makedirs(os.path.join(OUTPUT_PATH, "09_overlaps", "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "09_overlaps", "column_counts"), exist_ok=True)



# ==========================
# Load and save data
# ==========================
img_offset = 0
resolutions, img_counts = [], [0]

mcc, col = {}, {}

# Loop through movie subsets
for input_img in input_files:
    if not input_img.endswith(".ims"):
        continue

    mcc_temp, col_temp = {}, {}
    input_img_path = os.path.join(INPUT_PATH, input_img)

    full_data = utils.load_hdf5(input_img_path)

    # Extract data
    data = full_data["DataSet"]['children']["ResolutionLevel 0"]['children']
    
    # Calculate resolutions
    metadata = full_data["DataSetInfo"]["children"]["Image"]["attrs"]
    X_min = float(b"".join(metadata["ExtMin0"]).decode("utf-8"))
    X_max = float(b"".join(metadata["ExtMax0"]).decode("utf-8"))
    Y_min = float(b"".join(metadata["ExtMin1"]).decode("utf-8"))
    Y_max = float(b"".join(metadata["ExtMax1"]).decode("utf-8"))
    Z_min = float(b"".join(metadata["ExtMin2"]).decode("utf-8"))
    Z_max = float(b"".join(metadata["ExtMax2"]).decode("utf-8"))
    X = float(b"".join(metadata["X"]).decode("utf-8"))
    Y = float(b"".join(metadata["Y"]).decode("utf-8"))
    Z = float(b"".join(metadata["Z"]).decode("utf-8"))
    X_res = (X_max-X_min)/X
    Y_res = (Y_max-Y_min)/Y
    Z_res = (Z_max-Z_min)/Z
    resolutions.append((X_res, Y_res, Z_res))

    # Get sitk images from structure
    for tp in data:
        t = int(tp.split()[-1])
        mcc_temp[t], _ = utils.crop_zyx_to_nonzero(data[tp]['children']["Channel 1"]['children']["Data"]["data"])
        col_temp[t], _ = utils.crop_zyx_to_nonzero(data[tp]['children']["Channel 0"]['children']["Data"]["data"])
        mcc_temp[t] = sitk.GetImageFromArray(mcc_temp[t])
        col_temp[t] = sitk.GetImageFromArray(col_temp[t])
        mcc_temp[t].SetSpacing([X_res, Y_res, Z_res])
        col_temp[t].SetSpacing([X_res, Y_res, Z_res])

    # Save sequentially
    for t in mcc_temp:
        tx = t + img_offset
        index = f"{tx:02d}"

        mcc[tx] = mcc_temp[t]
        col[tx] = col_temp[t]

        sitk.WriteImage(mcc[tx], f"{OUTPUT_PATH}/00_full_images/mcc/mcc_{index}.nrrd")
        sitk.WriteImage(col[tx], f"{OUTPUT_PATH}/00_full_images/col/col_{index}.nrrd")
    
    img_offset += len(mcc_temp)
    img_counts.append(len(mcc))
    
# Save img_counts
img_counts.pop()
utils.save_variable(img_counts, os.path.join(OUTPUT_PATH, "00_full_images", "img_counts.pkl"))
    
# Save resolutions
utils.save_variable(resolutions, os.path.join(OUTPUT_PATH, "00_full_images", "org_resolutions.pkl"))

    
# ========================== 
# Info
# ==========================
print(f"Images to manually adjust: {img_counts}")
print("FINISHED")