# ========================== 
# Imports 
# ==========================
import os
import numpy as np
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
print("STEP 8: VORONOI REGIONS CALCULATION")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

# Image lists
center_list = os.listdir(f"{OUTPUT_PATH}/07_rotated_binary_images/centers")
center_list = sorted(center_list)

# Load
mcc, centers = {}, {}
for name_center in center_list:
    ts = int(name_center[20:22])
    
    centers[ts] = sitk.ReadImage(f"{OUTPUT_PATH}/07_rotated_binary_images/centers/{name_center}")
    

# ========================== 
# Compute voronoi regions
# ==========================
size = centers[0].GetSize()
spacing = centers[0].GetSpacing()

voronoi, voronoi_2d = {}, {}
for ts in tqdm(centers, desc="Computing Voronoi regions"):
    centers_2d = sitk.MaximumProjection(centers[ts], 2)     # Project centers in 2d
    
    distance_map = sitk.DanielssonDistanceMap(centers_2d, inputIsBinary=True, squaredDistance=False, useImageSpacing=False)    # Distance map
    V = sitk.MorphologicalWatershedFromMarkers(distance_map, centers_2d, markWatershedLine=False)                              # Watershed
    
    voronoi_2d[ts] = V
    
    # Remap voronoi labels to match center labels
    V_np = sitk.GetArrayViewFromImage(V).squeeze()

    # Stack voronoi and make it 3d
    V_3d_np = np.stack([V_np] * size[2], axis=0)
    V_3d = sitk.GetImageFromArray(V_3d_np)

    V_3d = sitk.Cast(V_3d, sitk.sitkUInt8)
    V_3d.SetSpacing(spacing)
    V_3d.SetOrigin(V.GetOrigin())
    V_3d.SetDirection((1,0,0, 0,1,0, 0,0,1))
    
    # Save into dict
    voronoi[ts] = V_3d
    

# ========================== 
# Save Voronoi images
# ==========================
import matplotlib.pyplot as plt
for ts in tqdm(voronoi, desc="Saving Voronoi images"):
    sitk.WriteImage(voronoi[ts], f"{OUTPUT_PATH}/08_voronoi_regions/voronoi/voronoi_{ts:02d}.nrrd")
    sitk.WriteImage(voronoi_2d[ts], f"{OUTPUT_PATH}/08_voronoi_regions/voronoi_2d/voronoi_2d_{ts:02d}.tif")


# ========================== 
# Info
# ==========================
print("FINISHED")