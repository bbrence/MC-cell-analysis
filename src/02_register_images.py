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
# Load variables 
# ==========================
OUTPUT_PATH = config["output_path"]

img_counts = utils.load_variable(os.path.join(OUTPUT_PATH, "00_full_images", "img_counts.pkl"))


# ========================== 
# Info
# ==========================
print("STEP 2: REGISTERING IMAGES")
print(f"Manually aligned images: {img_counts}")


# ========================== 
# Load images 
# ==========================
mcc_list = os.listdir(f"{OUTPUT_PATH}/01_resampled_images/mcc")
col_list = os.listdir(f"{OUTPUT_PATH}/01_resampled_images/col")

mcc, col = {}, {}
for name in mcc_list:
    idx = int(name[4:6])
    mcc[idx] = ants.image_read(f"{OUTPUT_PATH}/01_resampled_images/mcc/" + name)

for name in col_list:
    idx = int(name[4:6])
    col[idx] = ants.image_read(f"{OUTPUT_PATH}/01_resampled_images/col/" + name)

# Sort dict
mcc = dict(sorted(mcc.items()))
col = dict(sorted(col.items()))


# ========================== 
# Registration and calculate metrics
# ==========================
mcc_registered = mcc
col_registered = col

GCs = {}
transform_list = []

TRANSLATION_VECTOR_SET = config["02_register_images"]["translation_vector_set"]
translation_vector = TRANSLATION_VECTOR_SET[0]

for idx in mcc_registered:  
    if idx == 0:    # First img doesn't get registered
        print(f"Frame {idx} GC: 1")
        continue
    
    # Adjust for the translation vector
    if idx in img_counts:
        position = img_counts.index(idx)
        translation_vector = TRANSLATION_VECTOR_SET[position]

    mcc_registered[idx] = utils.translate_image(mcc_registered[idx], translation_vector)
    col_registered[idx] = utils.translate_image(col_registered[idx], translation_vector)

    # If normal image
    if idx not in img_counts:
        # Mcc
        mcc_registered[idx] = ants.apply_transforms(
            fixed=mcc_registered[0],
            moving=mcc_registered[idx],
            transformlist=transform_list
            )
        # Cols
        col_registered[idx] = ants.apply_transforms(
            fixed=col_registered[0], 
            moving=col_registered[idx], 
            transformlist=transform_list
            )

        registration = ants.registration(
            fixed=mcc_registered[idx-1],
            moving=mcc_registered[idx],
            type_of_transform='Similarity',
            initial_transform='Identity',
            aff_metric="GC",
            aff_sampling=256,
            aff_random_sampling_rate=0.1,
            aff_iterations=(5000, 2000, 1000, 10),
            aff_shrink_factors=(10, 4, 2, 1), 
            aff_smoothing_sigmas=(4, 2, 1, 0),
            use_legacy_histogram_matching=True,
            multivariate_extras=[("GC", col_registered[idx-1], col_registered[idx], 1.0, None)]
            )
    
    # If first in sequence
    else:
        registration = ants.registration(
            fixed=mcc_registered[idx-1],
            moving=mcc_registered[idx],
            type_of_transform='Similarity',
            initial_transform='Identity',
            aff_metric="GC",
            aff_sampling=256,
            aff_random_sampling_rate=1,
            # aff_iterations=(10000, 5000, 3000, 50),
            # aff_shrink_factors=(6, 4, 2, 1), 
            # aff_smoothing_sigmas=(3, 2, 1, 0),
            # aff_iterations=(100, 50, 20),
            # aff_shrink_factors=(2, 1, 1),
            # aff_smoothing_sigmas=(1, 0.5, 0),
            aff_iterations=(300, 100, 30),
            aff_shrink_factors=(3, 2, 1), 
            aff_smoothing_sigmas=(1, 0.5, 0),
            use_legacy_histogram_matching=True,
            )
        
        transform_list = []

    # Mcc
    mcc_registered[idx] = registration['warpedmovout']

    # Cols
    transform = registration['fwdtransforms'][0]
    col_registered[idx] = ants.apply_transforms(fixed=col_registered[0], moving=col_registered[idx], transformlist=[transform])
    
    # Append transformation
    transform_list.append(registration['fwdtransforms'][0])

    # GC calc and print
    GCs[idx] = utils.gradient_correlation(mcc_registered[idx-1], mcc_registered[idx])
    print(f"Frame {idx} GC: {GCs[idx]:.3f}")
    

# ========================== 
# Save registered images
# ==========================
for idx in tqdm(mcc_registered, desc="Saving"):
    index = f"{idx:02d}"

    ants.image_write(mcc_registered[idx].astype('uint8'), f"{OUTPUT_PATH}/02_registered_images/mcc/mcc_{index}.nrrd")
    ants.image_write(col_registered[idx].astype('uint8'), f"{OUTPUT_PATH}/02_registered_images/col/col_{index}.nrrd") 


# ========================== 
# Save metrics
# ==========================
utils.save_variable(GCs, os.path.join(OUTPUT_PATH, "02_registered_images/GCs.pkl"))

keys = list(GCs.keys())
GC_vals = list(GCs.values())

plt.plot(keys, GC_vals, label="GC", color="blue")

plt.xlabel("Step")
plt.ylabel("Metric Value")
plt.title("GC")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_PATH}/02_registered_images/GC_plot.png")


# ========================== 
# Info
# ==========================
print(f"Check the registration metrics!")
print("FINISHED")