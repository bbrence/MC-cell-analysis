# ========================== 
# Imports 
# ==========================
import os
import ants
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils


# ========================== 
# Load config 
# ==========================
config = utils.import_config()


# ========================== 
# Info
# ==========================
print("STEP 4: REGISTERING IMAGES - FINE REGISTRATION")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

mcc_list = os.listdir(f"{OUTPUT_PATH}/03_cropped_images/mcc")
col_list = os.listdir(f"{OUTPUT_PATH}/03_cropped_images/col")

mcc, col = {}, {}
for name in mcc_list:
    idx = int(name[4:6])
    mcc[idx] = ants.image_read(f"{OUTPUT_PATH}/03_cropped_images/mcc/" + name)

for name in col_list:
    idx = int(name[4:6])
    col[idx] = ants.image_read(f"{OUTPUT_PATH}/03_cropped_images/col/" + name)

# Sort dict
mcc = dict(sorted(mcc.items()))
col = dict(sorted(col.items()))


# ========================== 
# Create masks
# ==========================
MASKING = config["04_register_images_2"]["masking"]

masks = {}
if MASKING:
    for idx in mcc:
        masks[idx] = utils.threshold_numpy(mcc[idx], 5.0, 255.0)
        
        
# ========================== 
# Fine registration and calculate metrics
# ==========================
mcc_registered = mcc
col_registered = col

GCs = {}
transform_list = []

for idx in mcc_registered:
    if idx == 0:    # First img doesn't get registered
        print(f"Frame {idx} GC: 1")
        continue

    registration = ants.registration(
        fixed=mcc[idx-1],
        moving=mcc_registered[idx],
        mask=masks[idx-1],
        moving_mask=masks[idx],
        mask_all_stages=True,
        type_of_transform='Affine',
        initial_transform='Identity',
        aff_metric='GC',
        aff_sampling=256,
        aff_random_sampling_rate=0.25,
        # aff_iterations=50,
        # aff_shrink_factors=1,
        # aff_smoothing_sigmas=0,
        )
    
    # Mcc
    mcc_registered[idx] = registration['warpedmovout']
    
    # Col
    col_registered[idx] = ants.apply_transforms(fixed=col_registered[0], moving=col_registered[idx], transformlist=registration['fwdtransforms'])
    
    # GC calc and print
    GCs[idx] = utils.gradient_correlation(mcc_registered[idx-1], mcc_registered[idx])
    print(f"Frame {idx} GC: {GCs[idx]:.3f}")


# ========================== 
# Save registered images
# ==========================
for idx in tqdm(mcc_registered, desc="Saving"):
    index = f"{idx:02d}"

    ants.image_write(mcc_registered[idx].astype('uint8'), f"{OUTPUT_PATH}/04_registered_images_2/mcc/mcc_{index}.nrrd")
    ants.image_write(col_registered[idx].astype('uint8'), f"{OUTPUT_PATH}/04_registered_images_2/col/col_{index}.nrrd")


# ========================== 
# Save metrics
# ==========================
utils.save_variable(GCs, os.path.join(OUTPUT_PATH, "04_registered_images_2/GCs.pkl"))

keys = list(GCs.keys())
GC_vals = list(GCs.values())

plt.plot(keys, GC_vals, label="GC", color="blue")

plt.xlabel("Step")
plt.ylabel("Metric Value")
plt.title("GC")
plt.ylim((0, 1))
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_PATH}/04_registered_images_2/GC_plot.png")
    

# ========================== 
# Info
# ==========================
print("FINISHED")