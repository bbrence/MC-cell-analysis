# ========================== 
# Imports 
# ==========================
import os
import ants
import numpy as np
import matplotlib.pyplot as plt

import utils


# ========================== 
# Load config 
# ==========================
config = utils.import_config()


# ========================== 
# Info
# ==========================
print("STEP 6: CENTER CORECTION, ONLY COLUMN CHANNEL")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

col_list = os.listdir(f"{OUTPUT_PATH}/04_registered_images_2/col")

col = {}
for name in col_list:
    ts = int(name[4:6])
    
    col[ts] = ants.image_read(f"{OUTPUT_PATH}/04_registered_images_2/col/" + name)
    col[ts].set_origin((0, 0, 0))
    
# Sort
col = dict(sorted(col.items()))
    
# Load centers
centers = ants.image_read(f"{OUTPUT_PATH}/04_registered_images_2/centers.nrrd")
centers = ants.from_numpy(centers.numpy().astype('uint32'))
ants.copy_image_info(col[0], centers)


# ========================== 
# Preprocess centers
# ==========================
centers= utils.get_label_centroids_image(centers)


# ========================== 
# Create registration mask
# ==========================
MASK_DILATION_FACTOR = config["06_correct_centers"]["mask_dilation"]

centers_np = np.ascontiguousarray((centers.numpy() > 0).astype(np.uint32))
mask = ants.from_numpy(centers_np, origin=centers.origin, spacing=centers.spacing, direction=centers.direction)

spacing = col[ts].spacing
radii = utils.get_dilation_radii(spacing, MASK_DILATION_FACTOR)
mask = utils.anisotropic_dilate(mask, *radii)

# Save mask
ants.image_write(mask, f"{OUTPUT_PATH}/06_corrected_r7_centers/mask.nrrd")


# ========================== 
# Registration
# ==========================
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "24"

# Dilate denters so they do not disappear because of resampling after alignments
centers_dilated = utils.anisotropic_dilate(centers, 3, 3, 3)

# Take each n-th image for alignment
STEP = config["06_correct_centers"]["step"]
time_steps = list(col.keys())

# Align to first or previous time step
ALIGN_TO_FIRST = config["06_correct_centers"]["align_to_first"]

ts_to_register = time_steps[::STEP] + ([time_steps[-1]] if time_steps[-1] not in time_steps[::STEP] else [])
print(f"Indices to register: {ts_to_register}")

GC = {}
centroids = {}
col_registered = {}

previous_ts = 0
for ts in col:
    if ts not in ts_to_register:
        continue
    
    if ts == 0:
        col_registered[0] = col[0]
        centroids[0] = utils.get_label_centroids(centers_dilated)
        
        centers_dilated_previous = centers_dilated.clone()
        centers_dilated_first = centers_dilated.clone()
        
        print(f"Frame 0 GC: 1")
        
        ants.image_write(col[0].clone(pixeltype="uint8"), f"{OUTPUT_PATH}/06_corrected_r7_centers/col/col_registered_00.nrrd")
        ants.image_write(centers.clone(pixeltype="uint8"), f"{OUTPUT_PATH}/06_corrected_r7_centers/centers/col_centers_00.nrrd")
        continue
    
    
    T = 0 if ALIGN_TO_FIRST else previous_ts
    
    registration_elastic = ants.registration(fixed=col_registered[T],
                                                moving=col[ts],
                                                mask=mask,          
                                                moving_mask=mask,
                                                mask_all_stages=True,                          
                                                type_of_transform="SyN",
                                                initial_transform="Identity",
                                                syn_metric="CC",
                                                syn_sampling=32,
                                                reg_iterations=(150, 0, 0),
                                                # reg_iterations=(1, 0, 0),     # Uncomment for testing to speed up registration
                                                verbose=False  
                                                )
    
    col_registered[ts] = registration_elastic['warpedmovout']
    
    centers_ref = centers_dilated_first if ALIGN_TO_FIRST else centers_dilated_previous
    centers_trans = ants.apply_transforms(fixed=centers_ref, 
                                          moving=centers_ref, 
                                          transformlist=registration_elastic["invtransforms"], 
                                          interpolator="nearestNeighbor")
    
    if not ALIGN_TO_FIRST:
        centers_dilated_previous = centers_trans.clone()
    
    centers_trans = utils.get_label_centroids_image(centers_trans)
    centroids[ts] = utils.get_label_centroids(centers_trans)
    
    GC[ts] = utils.gradient_correlation(col_registered[T], col_registered[ts])
    print(f"Frame {ts} GC: {GC[ts]:.3f}")
    
    previous_ts = ts
    
    ants.image_write(col_registered[ts].clone(pixeltype="uint8"), f"{OUTPUT_PATH}/06_corrected_r7_centers/col/col_registered_{ts:02d}.nrrd")
    ants.image_write(centers_trans.clone(pixeltype="uint8"), f"{OUTPUT_PATH}/06_corrected_r7_centers/centers/col_centers_{ts:02d}.nrrd")
    

# ========================== 
# Save metrics
# ==========================
utils.save_variable(GC, os.path.join(OUTPUT_PATH, "06_corrected_r7_centers/GCs.pkl"))

keys = list(GC.keys())
GC_vals = list(GC.values())

plt.plot(keys, GC_vals, label="GC", color="blue")

plt.xlabel("Step")
plt.ylabel("Metric Value")
plt.title("GC")
plt.ylim((0, 1))
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_PATH}/06_corrected_r7_centers/GC_plot.png")


# ========================== 
# Interpolation for images that were not registered
# ==========================
interpolated_ts = []
for i, _ in enumerate(ts_to_register[:-1]):
    ts_to_interpolate = np.arange(ts_to_register[i]+1, ts_to_register[i+1])
    interpolated_ts.extend(ts_to_interpolate.tolist())
    n_interpolation_points = len(ts_to_interpolate)
    
    centroids_A = centroids[ts_to_register[i]]
    centroids_B = centroids[ts_to_register[i+1]]
    
    # Sometimes a centroid is missing in following time step due to registration
    common_centroids = []
    for idx in centroids_B:
        if idx in centroids_A:
            common_centroids.append(idx)
        else:
            print(f"Time step {ts_to_register[i+1]} does not contain photoreceptor center {idx}!")
            
    for ts in ts_to_interpolate:
        centroids[ts] = {}
        
    # Iterate over common centroids and interpolate inbetween
    for idx in common_centroids:
        interpolated_centroids = np.linspace(centroids_A[idx], centroids_B[idx], n_interpolation_points+2)[1:-1] # Remove first and last since they are centroids_A[idx] and centroids_B[idx]
        
        for ts, pt in zip(ts_to_interpolate, interpolated_centroids):
            centroids[ts][idx] = tuple(pt)
    
    centroids = dict(sorted(centroids.items()))


# ========================== 
# Save interpolated centers as images
# ==========================
for ts in centroids:
    if ts not in interpolated_ts:   # Skip images that were registered
        continue
    
    centers_interpolated = utils.centroid_coords_to_image(centroids[ts], centers)
    ants.image_write(centers_interpolated.clone(pixeltype="uint8"), f"{OUTPUT_PATH}/06_corrected_r7_centers/centers/col_centers_{ts:02d}.nrrd")

# Save which indices were interpolated as pkl
utils.save_variable(interpolated_ts, os.path.join(OUTPUT_PATH, "06_corrected_r7_centers/interpolated_ts.pkl"))


# ========================== 
# Save centroids as pkl and JSON
# ==========================
utils.save_variable(centroids, os.path.join(OUTPUT_PATH, "06_corrected_r7_centers/centroids.pkl"))
utils.save_variable_json(centroids, os.path.join(OUTPUT_PATH, "06_corrected_r7_centers/centroids.json"))


# ========================== 
# Info
# ==========================
print(f"Interpolated centers: {interpolated_ts}")
print("FINISHED")