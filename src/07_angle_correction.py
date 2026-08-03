# ========================== 
# Imports 
# ==========================
import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from sklearn.decomposition import PCA

import utils


# ========================== 
# Load config 
# ==========================
config = utils.import_config()


# ========================== 
# Info
# ==========================
print("STEP 7: ANGLE CORECTION")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

# Image lists
mcc_list = os.listdir(f"{OUTPUT_PATH}/05_thresholded_images/mcc")
col_list = os.listdir(f"{OUTPUT_PATH}/04_registered_images_2/col")
center_list = os.listdir(f"{OUTPUT_PATH}/06_corrected_r7_centers/centers")

# Sort lists
mcc_list, col_list, center_list = sorted(mcc_list), sorted(col_list), sorted(center_list)

# Load
mcc, col, centers = {}, {}, {}
for name_mcc, name_col, name_center in zip(mcc_list, col_list, center_list):
    ts = int(name_mcc[4:6])
    
    mcc[ts] = sitk.ReadImage(f"{OUTPUT_PATH}/05_thresholded_images/mcc/{name_mcc}")
    col[ts] = sitk.ReadImage(f"{OUTPUT_PATH}/04_registered_images_2/col/{name_col}")
    centers[ts] = sitk.ReadImage(f"{OUTPUT_PATH}/06_corrected_r7_centers/centers/{name_center}")
    
    mcc[ts].SetOrigin((0, 0, 0))
    mcc[ts].SetSpacing(centers[ts].GetSpacing())    # Spacing might have been compromised during manual corrections of the thresholded images
    
    col[ts].SetOrigin((0, 0, 0))

    
# ========================== 
# Resample images to isotropic 
# ==========================
for ts in tqdm(mcc, desc="Resampling"):
    mcc[ts] = utils.resample_to_isotropic(mcc[ts])
    
    col[ts] = utils.resample_to_isotropic(col[ts], sitk.sitkLinear)
    
    centers[ts] = utils.resample_to_isotropic(centers[ts])
    centers[ts] = utils.centers_to_single_voxels(centers[ts])

# ========================== 
# Save voxel spacings
# ==========================
voxel_spacings = mcc[0].GetSpacing()
utils.save_variable(voxel_spacings, f"{OUTPUT_PATH}/07_rotated_binary_images/voxel_spacings.pkl")

# ========================== 
# Determine rotation parameters
# ==========================
METHOD = config["07_angle_correction"]["method"]

mcc_first = mcc[0]    # Rotation parameters based on first image
mcc_first_np = sitk.GetArrayFromImage(mcc_first)
spacing = np.array(mcc_first.GetSpacing())

# Get the coordinates of the object's voxels.
mcc_first_point_cloud = np.argwhere(mcc_first_np == 1)[:, ::-1]    # Make x, y, z

# Define the target axis you want to align with (the Z-axis)
target_axis = np.array([0.0, 0.0, 1.0])

# Plane fitting
normal = np.array([])
mcc_first_centroid = []

if METHOD == "pca":
    # Perform PCA to find the best-fit plane
    pca = PCA(n_components=2)
    pca.fit(mcc_first_point_cloud)
    components = pca.components_

    # Get the normal vector to the best-fit plane.
    # Since the input was (x, y, z), the output is also (x, y, z)
    normal = np.cross(components[0], components[1])
    normal = normal / np.linalg.norm(normal)    # Normalize the vector

    # Get the centroid of the data in (x, y, z) coordinates
    mcc_first_centroid = pca.mean_
    mcc_first_centroid = [int(i) for i in mcc_first_centroid]

elif METHOD == "ransac":
    # RANSAC plane fit
    DISTANCE_TH = config["07_angle_correction"]["distance_threshold"]
    N_ITER = config["07_angle_correction"]["n_iter"]
    
    # Run ransac
    normal, d, inliers = utils.ransac_plane(mcc_first_point_cloud, n_iterations=N_ITER, distance_threshold=DISTANCE_TH, print_output=False)

    # Centroid: mean of inlier points only
    mcc_first_centroid_float = mcc_first_point_cloud[inliers].mean(axis=0)
    mcc_first_centroid = [int(i) for i in mcc_first_centroid_float]

# Flip if pointing away from z-axis
if np.dot(normal, target_axis) < 0:
    normal = -normal

# The rotation axis is perpendicular to both the normal and target axes.
rotation_axis = np.cross(target_axis, normal)
rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

# The rotation angle is the angle between the normal and target axes.
rotation_angle = np.arccos(np.clip(np.dot(normal, target_axis), -1.0, 1.0))


# ========================== 
# Create versor transform and resampler
# ==========================
transform = sitk.VersorTransform()
transform.SetRotation(rotation_axis, rotation_angle)
transform.SetCenter(mcc_first.TransformIndexToPhysicalPoint(mcc_first_centroid))

resampler = sitk.ResampleImageFilter()
resampler.SetTransform(transform)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0.0)
resampler.SetReferenceImage(mcc_first)


# ========================== 
# Apply transform to Mcc images
# ==========================
mcc_rotated = {}
for ts in tqdm(mcc, desc="Adjusting angles of Mcc"):
    mcc_rotated[ts] = resampler.Execute(mcc[ts])

    # Remove small parts
    cc_filter = sitk.ConnectedComponentImageFilter()
    mcc_rotated[ts] = cc_filter.Execute(mcc_rotated[ts])

    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SortByObjectSizeOn()
    mcc_rotated[ts] = relabel_filter.Execute(mcc_rotated[ts])

    mcc_rotated[ts] = sitk.BinaryThreshold(mcc_rotated[ts], lowerThreshold=1, upperThreshold=1)
    

# ========================== 
# Apply transform to Col images
# ==========================
col_rotated = {}
for ts in tqdm(col, desc="Adjusting angles of Col"):
    col_rotated[ts] = resampler.Execute(col[ts])
    
# ==========================
# Save centroid and normal as pkl
# ==========================
cn = {"normal": normal, "centroid": mcc_first_centroid}
utils.save_variable(cn, f"{OUTPUT_PATH}/07_rotated_binary_images/centroid_normal.pkl")


# ==========================
# Save rotated images
# ==========================
for ts in tqdm(mcc_rotated, desc="Saving Mcc and col images"):
    sitk.WriteImage(mcc_rotated[ts], f"{OUTPUT_PATH}/07_rotated_binary_images/mcc/mcc_binary_rotated_{ts:02d}.nrrd")
    sitk.WriteImage(col_rotated[ts], f"{OUTPUT_PATH}/07_rotated_binary_images/col/col_binary_rotated_{ts:02d}.nrrd")
    
    
# ==========================
# Save projected Mcc cell
# ==========================
for ts in tqdm(mcc_rotated, desc="Saving Mcc projection"):
    mcc_proj = sitk.MaximumProjection(sitk.BinaryErode(mcc[ts], [1, 1, 1]), 2)
    sitk.WriteImage(mcc_proj, f"{OUTPUT_PATH}/07_rotated_binary_images/mcc_2d_proj/mcc_2d_{ts:02d}.tif")

# ==========================
# Project center points to same plane as Mcc
# ==========================
# Plane parameters for point projection
a, b, c = normal
x0, y0, z0 = mcc_first_centroid

for ts in tqdm(centers, desc="Adjusting column centers"):
    centers_arr = sitk.GetArrayFromImage(centers[ts])
    
    coords = np.argwhere(centers_arr > 0)
    values = centers_arr[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    coords_and_values = np.column_stack((coords, values))
    
    # Get projected Z values for centers
    X = coords_and_values.T[2]
    Y = coords_and_values.T[1]
    Z = np.round(z0 - (a*(X-x0) + b*(Y-y0))/c)
    
    # Projected points
    coords_and_values_projected = coords_and_values.copy()
    coords_and_values_projected[:, 0] = Z
    
    # Get image and prepare for transform
    centers_image_projected = utils.get_sitk_image_from_point_cloud_with_values(coords_and_values_projected, mcc_first.GetSize())
    centers_image_projected.SetSpacing(mcc_first.GetSpacing())
    centers_image_projected = sitk.GrayscaleDilate(centers_image_projected, [10, 10, 10])
    
    # Transform
    centers_rotated = resampler.Execute(centers_image_projected)
    centers_rotated = utils.centers_to_single_voxels(centers_rotated)
    
    # Save center image
    sitk.WriteImage(centers_rotated, f"{OUTPUT_PATH}/07_rotated_binary_images/centers/col_centers_rotated_{ts:02d}.nrrd")


# ========================== 
# Info
# ==========================
print("FINISHED")