# ========================== 
# Imports 
# ==========================
import os
import cv2
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import utils


# ========================== 
# Load config 
# ==========================
config = utils.import_config()


# ========================== 
# Info
# ==========================
print("STEP 9: OVERLAP CALCULATION")


# ========================== 
# Load images 
# ==========================
OUTPUT_PATH = config["output_path"]

# Image lists
mcc_list = os.listdir(f"{OUTPUT_PATH}/07_rotated_binary_images/mcc")
voronoi_list = os.listdir(f"{OUTPUT_PATH}/08_voronoi_regions/voronoi")

# Sort lists
mcc_list, voronoi_list = sorted(mcc_list), sorted(voronoi_list)

# Load
mcc, voronoi = {}, {}
for name_mcc, name_voronoi in zip(mcc_list, voronoi_list):
    ts = int(name_mcc[19:21])
    
    mcc[ts] = sitk.ReadImage(f"{OUTPUT_PATH}/07_rotated_binary_images/mcc/{name_mcc}")
    voronoi[ts] = sitk.ReadImage(f"{OUTPUT_PATH}/08_voronoi_regions/voronoi/{name_voronoi}")
    
    
# ========================== 
# Compute overlaps of Mcc cells and Voronoi regions
# ==========================
overlaps = {}
for ts in tqdm(mcc, desc="Computing overlaps"):
    mcc_np = sitk.GetArrayViewFromImage(mcc[ts])
    V_np = sitk.GetArrayViewFromImage(voronoi[ts])

    mask = (mcc_np == 1)

    # Count occurrences of labels only where mask is True
    counts = np.bincount(V_np[mask].astype(np.int64))

    overlaps[ts] = {int(label): int(count) for label, count in enumerate(counts) if count > 0}
        

# ========================== 
# Save overlaps
# ==========================
utils.save_variable(overlaps, f"{OUTPUT_PATH}/09_overlaps/overlaps.pkl")
utils.save_variable_json(overlaps, f"{OUTPUT_PATH}/09_overlaps/overlaps.json")


# ========================== 
# Get volumes and normalize overlaps
# ==========================
SAVE_OVERLAPS_NORMALIZED = config["09_compute_overlaps"]["save_overlaps_normalized"]

if SAVE_OVERLAPS_NORMALIZED:
    volumes, overlaps_normalized = {}, {}
    
    # Volumes
    for ts in voronoi:
        volumes[ts] = utils.get_label_volumes(voronoi[ts])
    
    utils.save_variable(volumes, f"{OUTPUT_PATH}/09_overlaps/voronoi_volumes.pkl")          # Save volumes
    utils.save_variable_json(volumes, f"{OUTPUT_PATH}/09_overlaps/voronoi_volumes.json")
    
    # Normalized overlaps
    for ts in overlaps:
        overlaps_normalized[ts] = {label: overlaps[ts][label]/volumes[ts][label] for label in overlaps[ts]}
    
    utils.save_variable(overlaps_normalized, f"{OUTPUT_PATH}/09_overlaps/overlaps_normalized.pkl")          # Save normalized overlaps
    utils.save_variable_json(overlaps_normalized, f"{OUTPUT_PATH}/09_overlaps/overlaps_normalized.json")


# ========================== 
# Plot heatmaps
# ==========================
PLOT_HEATMAPS = config["09_compute_overlaps"]["plot_heatmaps"]
HEATMAPS_NORMALIZED = config["09_compute_overlaps"]["heatmaps_normalized"]
COLUMN_SEPARATION = config["09_compute_overlaps"]["column_separation"]

if PLOT_HEATMAPS:
    # Check if heatmap plot should be normalized
    overlaps_heatmap = overlaps_normalized if HEATMAPS_NORMALIZED else overlaps

    # Loop through time steps
    for ts in tqdm(overlaps_heatmap, desc="Saving heatmaps"):
        mcc_eroded = sitk.MaximumProjection(sitk.BinaryErode(mcc[ts], [1, 1, 1]), 2)
        mcc_np = sitk.GetArrayFromImage(mcc_eroded).squeeze()
        
        voronoi_np = sitk.GetArrayFromImage(voronoi[ts])
        voronoi_np_2d = voronoi_np[0, :, :]
        
        # Create heat image
        heatmap_np = np.zeros(mcc_np.shape)
        for x in range(heatmap_np.shape[1]):
            for y in range(heatmap_np.shape[0]):
                    label = int(voronoi_np_2d[y, x])            # Get label value at that coordinate
                    
                    # If overlap is not 0
                    if label in overlaps_heatmap[ts]:
                        count = overlaps_heatmap[ts][label]     # Check what is count at this coordinate
                        heatmap_np[y, x] = count                # Add count value to same coordinates
                    else:
                        heatmap_np[y, x] = 0
        
        # Separation between Voronoi cells
        if COLUMN_SEPARATION:
            edges = cv2.Canny(voronoi_np_2d.astype(np.uint8), threshold1=0.5, threshold2=1.0)
            edges = np.array(edges, dtype=np.uint8)
            
            heatmap_np[edges > 0] = np.nan

        # Find region centers
        labels = np.unique(voronoi_np_2d)
        labels = labels[labels != 0]                # Exclude background
        centroids = center_of_mass(voronoi_np_2d, labels=voronoi_np, index=labels)
        
        # Create the figure
        fig = go.Figure()
        
        # Add heatmape to figure
        fig.add_trace(go.Heatmap(
            z = heatmap_np,
            # colorscale = 'Viridis',
            showscale = False,
            zmin = heatmap_np.min(),
            zmax = heatmap_np.max()
        ))
        
        # Add cell to figure
        fig.add_trace(go.Heatmap(
            z = mcc_np,
            colorscale = [[0, 'rgba(0,0,0,0)'],         # Transparent where 0
                          [1, 'rgba(255,0,0,0.4)']],    # Semi-transparent red where 1
            showscale = False,
            zmin = 0,
            zmax = 1,
        ))
        
        # Add label annotations
        for label, (y, x) in zip(labels, centroids):
            fig.add_annotation(
                x = x,
                y = y,
                text = str(label),
                showarrow = False,
                font = dict(color='white', size=16),
                xanchor = 'center',
                yanchor = 'middle',
                xref = 'x',
                yref = 'y'
            )
            
        # Flip y-axis to match image coordinates
        fig.update_yaxes(autorange='reversed')
        
        # Hide axes
        fig.update_layout(
            autosize = False,
            margin = dict(l=0, r=0, t=0, b=0),
            paper_bgcolor = 'white',
            plot_bgcolor = 'white',
            xaxis = dict(showticklabels=False, showgrid=False, 
                        zeroline=False, showline=False, 
                        ticks='', scaleanchor='y', 
                        constrain='domain'),
            yaxis = dict(showticklabels=False, showgrid=False, 
                        zeroline=False, showline=False, 
                        ticks='', scaleanchor='x', 
                        constrain='domain', autorange='reversed')
        )
        
        # Save as png
        shape = mcc[ts].GetSize()
        suffix = "normalized_" if HEATMAPS_NORMALIZED else ""
        
        fig.write_image(f"{OUTPUT_PATH}/09_overlaps/heatmaps/heatmap_{suffix}{ts:02d}.png", width=shape[0], height=shape[1], scale=3)
            

# ========================== 
# Plot column counts
# ==========================
PLOT_COLUMN_COUNTS = config["09_compute_overlaps"]["plot_column_counts"]
COLUMN_COUNTS_NORMALIZED = config["09_compute_overlaps"]["column_counts_normalized"]

percentage = 100 if COLUMN_COUNTS_NORMALIZED else 1

if PLOT_COLUMN_COUNTS:
    overlaps_cc = overlaps_normalized if COLUMN_COUNTS_NORMALIZED else overlaps
    
    max_count = 0
    for ts, vec in overlaps_cc.items():
        for col, count in vec.items():
            if max_count < count:
                max_count = count

    rounding_factor = 10 if COLUMN_COUNTS_NORMALIZED else 10000
    max_count = utils.round_up(max_count*percentage, rounding_factor)

    col_list = np.unique(sitk.GetArrayViewFromImage(voronoi[0]))    # List of all columns

    for col in tqdm(col_list, desc="Saving column plots"):        
        time_steps = list(overlaps_cc.keys())
        counts = [overlaps_cc[ts][col]*percentage if col in overlaps_cc[ts] else 0 for ts in time_steps]
        
        if all(c == 0 for c in counts):
            continue

        plt.figure(figsize=(6, 4))
        plt.bar(time_steps, counts, width=0.8, color='skyblue')
        plt.ylim([0, max_count])
        plt.xlabel('Time step', fontsize=14)
        plt.ylabel('Overlap count [%]', fontsize=14)
        plt.title(f'Column {col}', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        suffix = "normalized_" if COLUMN_COUNTS_NORMALIZED else ""
        plt.savefig(f"{OUTPUT_PATH}/09_overlaps/column_counts/cc_{suffix}{col:02d}.png")
        plt.close()


# ========================== 
# Depth plots
# ==========================
PLOT_DISTR = config["09_compute_overlaps"]["plot_distr"]

centroid = utils.load_variable(f"{OUTPUT_PATH}/07_rotated_binary_images/centroid_normal.pkl")["centroid"]
Z_plane = centroid[2]

# Get the distributions for each column in each time step
distributions = {}
for ts in tqdm(mcc, desc="Calculating distributions"):
    mcc_np = sitk.GetArrayViewFromImage(mcc[ts])
    V_np = sitk.GetArrayViewFromImage(voronoi[ts])
    
    counts_per_label_per_z = {}
    for z in range(V_np.shape[0]):
        mask_z = (mcc_np[z] == 1)
        labels_z = V_np[z][mask_z].astype(np.int64)

        binc = np.bincount(labels_z)

        for label, count in enumerate(binc):
            if count == 0:
                continue

            if label not in counts_per_label_per_z:
                counts_per_label_per_z[label] = np.zeros(V_np.shape[0], dtype=int)

            counts_per_label_per_z[label][z] = count
    
    distributions[ts] = counts_per_label_per_z
    
utils.save_variable_json(distributions, f"{OUTPUT_PATH}/09_overlaps/distributions.json")

# Plot
if PLOT_DISTR:  
    for ts in tqdm(distributions, desc="Saving distributions"):
        # The vertical axis
        Z = V_np.shape[0]
        z_axis = np.arange(Z)

        # All labels present at time step
        labels = sorted(distributions[ts].keys())

        # Subplot shape
        n = len(labels)
        cols = 8
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, label in enumerate(labels):
            ax = axes[i]
            counts = distributions[ts][label]

            # Horizontal plot: counts on X, Z on Y
            ax.plot(counts, z_axis, color='steelblue')
            ax.axhline(y=Z_plane, color='red', linestyle='-', linewidth=1, label="Mcc cell plane")
            ax.set_title(f"Column {label}", fontsize=10)

        # Hide unused subplots
        # for j in range(i + 1, len(axes)):
        #     axes[j].axis("off")

        fig.supxlabel("Voxel count", fontsize=14)
        # fig.supylabel("Z slice", fontsize=14)
        fig.suptitle(f"Z-distribution per Voronoi region for time step {ts}", fontsize=16)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PATH}/09_overlaps/distributions/distr_{ts:02d}.png")
        plt.close()


# ========================== 
# Info
# ==========================
print("FINISHED")
