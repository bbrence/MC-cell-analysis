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
print("STEP 5: THRESHOLDING, ONLY MCC CHANNEL")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

mcc_list = os.listdir(f"{OUTPUT_PATH}/04_registered_images_2/mcc")

mcc = {}
for name in mcc_list:
    idx = int(name[4:6])
    mcc[idx] = sitk.ReadImage(f"{OUTPUT_PATH}/04_registered_images_2/mcc/" + name)
    
    
# ========================== 
# Normalize images
# ==========================
NORMALIZE = config["05_thresholding"]["normalize"]

if NORMALIZE:
    for idx in tqdm(mcc, desc="Normalizing"):
        mcc[idx] = sitk.RescaleIntensity(mcc[idx])
        

# ========================== 
# Filter images
# ==========================
FILTER = config["05_thresholding"]["filter"]
FILTER_RADIUS = config["05_thresholding"]["filter_radius"]

if FILTER:
    for idx in tqdm(mcc, desc="Filtering"):
        mcc[idx] = sitk.Median(mcc[idx], [FILTER_RADIUS, FILTER_RADIUS, FILTER_RADIUS])


# ========================== 
# Determine thresholds
# ==========================
OTSU_THRESHOLDING = config["05_thresholding"]["use_otsu"]

thresholds = {}
if OTSU_THRESHOLDING:
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    
    for idx in mcc:
        otsu.Execute(mcc[idx])
        thresholds[idx] = otsu.GetThreshold()
    
else:
    MANUAL_TH_VALUE = config["05_thresholding"]["manual_threshold"]
    thresholds = {idx: MANUAL_TH_VALUE for idx in mcc}


# ========================== 
# Perform thresholding
# ==========================
mcc_binary = {}
for idx in tqdm(mcc, desc="Thresholding"):
    mcc_binary[idx] = sitk.BinaryThreshold(mcc[idx], lowerThreshold=thresholds[idx])


# ========================== 
# Connected components and size filter
# ==========================
MIN_SIZE = config["05_thresholding"]["min_size"]
ONLY_KEEP_LARGEST = config["05_thresholding"]["only_keep_largest"]

mcc_out = {}
for idx in mcc_binary:
    # Connected components
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.FullyConnectedOff()
    mcc_cc = cc_filter.Execute(mcc_binary[idx])

    # Relabel
    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(MIN_SIZE)      # Remove smaller components
    relabeled_img = relabel.Execute(mcc_cc)

    # Threshold
    upper = 1 if ONLY_KEEP_LARGEST else relabel.GetNumberOfObjects()
    mcc_out[idx] = sitk.BinaryThreshold(relabeled_img, 1, upper)
    mcc_out[idx].CopyInformation(mcc_binary[idx])
    
    
# ========================== 
# Save images
# ==========================   
for idx in tqdm(mcc, desc="Saving"):
    index = f"{idx:02d}"
    sitk.WriteImage(mcc_out[idx], f"{OUTPUT_PATH}/05_thresholded_images/mcc/mcc_{index}.nrrd")
    
    
# ========================== 
# Info
# ==========================
if NORMALIZE:
    print(f"Images normalized")

if FILTER:
    print(f"Images filtered")
    
if OTSU_THRESHOLDING:
    print(f"Otsu thresholds: {thresholds}")
    
print("FINISHED")