# Column Explorer
# Heatmap + Clustering Explorer

import os
import json
import pickle
from glob import glob

import numpy as np
import pandas as pd
import panel as pn
import tifffile as tiff
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.ndimage import binary_dilation
from bokeh.events import MouseWheel, Tap
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    ColorBar,
    LabelSet,
    Span,
    HoverTool,
    Range1d,
    FixedTicker,
    BasicTicker,
    PrintfTickFormatter,
    NumeralTickFormatter,
    AllLabels,
)
from bokeh.palettes import Inferno256, Viridis256, Magma256, Plasma256, Turbo256

pn.extension(sizing_mode="stretch_width")


# =========================================================
# CONFIG
# =========================================================
DEFAULT_BASE_DIR = ""

INIT_STEP = 0
INIT_MODE = "normalized"
INIT_ALPHA = 0.6
INIT_CMAP = "Inferno"
INIT_VIEW_MODE = "Heatmap"

INIT_N_CLUSTERS = 3
INIT_CLUSTER_LINKAGE = "ward"
INIT_CLUSTER_METRIC = "euclidean"
INIT_INCLUDE_ZERO_COLUMNS = False
INIT_SHOW_MEAN_PATTERN = True
INIT_USE_PHYSICAL_Z_AXIS = False

DEFAULT_Z_UNIT = 1.0

ALL_CLUSTER_FEATURES = [
    "mean",
    "early_mean",
    "mid_mean",
    "late_mean",
    "std",
    "skewness",
    "kurtosis",
    "max",
    "nonzero_ratio",
    "early_nonzero_ratio",
    "mid_nonzero_ratio",
    "late_nonzero_ratio",
]

INIT_CLUSTER_FEATURES = [
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "max",
    "nonzero_ratio",
]

PALETTES = {
    "Inferno": Inferno256,
    "Viridis": Viridis256,
    "Magma": Magma256,
    "Plasma": Plasma256,
    "Turbo": Turbo256,
}

CLUSTER_COLORS = [
    "#FFFFFF",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

CLUSTER_LINKAGE_OPTIONS = ["ward", "complete", "average", "single"]
CLUSTER_METRIC_OPTIONS = ["euclidean", "manhattan", "cosine"]

MAIN_FIG_WIDTH = 1050
MAIN_FIG_HEIGHT = 650

DETAIL_FIG_WIDTH = 500
DETAIL_FIG_HEIGHT = 280

SECTION_WIDTH = 1120
TOP_SECTION_HEIGHT = 860

COMPARE_FIG_WIDTH = 560
COMPARE_FIG_HEIGHT = 300
COMPARE_PANEL_WIDTH = 620

RIGHT_CLUSTER_PANEL_WIDTH = 1120
MAIN_CONTENT_WIDTH = SECTION_WIDTH + COMPARE_PANEL_WIDTH + RIGHT_CLUSTER_PANEL_WIDTH + 40

EMPTY_PANEL_HEIGHT = 180
ROW_GAP = 20


# =========================================================
# PATH / IO HELPERS
# =========================================================
def build_paths(base_dir):
    return {
        "basis": os.path.join(base_dir, "07_rotated_binary_images", "mcc_2d_proj"),
        "voronoi": os.path.join(base_dir, "08_voronoi_regions", "voronoi_2d"),
        "raw_json": os.path.join(base_dir, "09_overlaps", "overlaps.json"),
        "norm_json": os.path.join(base_dir, "09_overlaps", "overlaps_normalized.json"),
        "dist_json": os.path.join(base_dir, "09_overlaps", "distributions.json"),
        "centroid_pkl": os.path.join(base_dir, "07_rotated_binary_images", "centroid_normal.pkl"),
        "voxel_spacings_pkl": os.path.join(base_dir, "07_rotated_binary_images", "voxel_spacings.pkl"),
        "voronoi_volumes_json": os.path.join(base_dir, "09_overlaps", "voronoi_volumes.json"),
    }


def get_sorted_tiff_files(folder):
    files = sorted(glob(os.path.join(folder, "*.tif")))
    if not files:
        raise ValueError(f"No TIFF files found in: {folder}")
    return files


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_z_unit(path):
    if not os.path.exists(path):
        return DEFAULT_Z_UNIT

    with open(path, "rb") as f:
        voxel_spacings = pickle.load(f)

    return float(voxel_spacings[2])


def load_step_images(step_idx, basis_files, voronoi_files):
    basis_img = tiff.imread(basis_files[step_idx])
    voronoi_img = tiff.imread(voronoi_files[step_idx])

    if basis_img.ndim != 2:
        raise ValueError(f"Basis image at step {step_idx} is not 2D. Shape: {basis_img.shape}")
    if voronoi_img.ndim != 2:
        raise ValueError(f"Voronoi image at step {step_idx} is not 2D. Shape: {voronoi_img.shape}")
    if basis_img.shape != voronoi_img.shape:
        raise ValueError(
            f"Shape mismatch at step {step_idx}: basis={basis_img.shape}, voronoi={voronoi_img.shape}"
        )

    return basis_img.astype(np.float64), voronoi_img.astype(np.int32)


def collect_all_region_ids_from_voronoi_files(voronoi_files):
    all_region_ids = set()

    for path in voronoi_files:
        voronoi_img = tiff.imread(path)

        if voronoi_img.ndim != 2:
            raise ValueError(f"Voronoi image is not 2D: {path}")

        ids = np.unique(voronoi_img)
        ids = ids[ids != 0]
        all_region_ids.update(int(x) for x in ids)

    return sorted(all_region_ids)


def parse_compare_column_ids():
    raw = compare_column_ids_input.value.strip()

    if not raw:
        return []

    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue

    unique_ids = []
    seen = set()
    for rid in ids:
        if rid not in seen:
            unique_ids.append(rid)
            seen.add(rid)

    return unique_ids


# =========================================================
# REGION / IMAGE HELPERS
# =========================================================
def create_heatmap(voronoi_img, step_values):
    heatmap = np.zeros_like(voronoi_img, dtype=np.float64)

    for rid in np.unique(voronoi_img):
        if rid == 0:
            continue
        rid_str = str(int(rid))
        if rid_str in step_values:
            heatmap[voronoi_img == rid] = float(step_values[rid_str])

    return heatmap


def normalize01(img):
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def get_region_id(x, y, voronoi_img):
    xi = int(round(x))
    yi = int(round(y))

    if xi < 0 or yi < 0 or yi >= voronoi_img.shape[0] or xi >= voronoi_img.shape[1]:
        return None

    return int(voronoi_img[yi, xi])


def build_time_series(region_id, json_data, step_range=None):
    rid_str = str(region_id)
    steps = sorted(json_data.keys(), key=lambda k: int(k))

    if step_range is not None:
        start, end = step_range
        steps = [s for s in steps if start <= int(s) <= end]

    times = np.array([int(s) for s in steps], dtype=np.int32)
    values = np.array(
        [float(json_data[s].get(rid_str, 0.0)) for s in steps],
        dtype=np.float64,
    )

    return times, values


def build_distribution(region_id, dist_data, step_idx):
    step_key = str(step_idx)
    rid_str = str(region_id)
    vals = dist_data.get(step_key, {}).get(rid_str, [])
    return np.array(vals, dtype=np.float64)

def get_column_volume(step_idx, column_id):
    volumes = data_store.get("voronoi_volumes", {})

    step_entry = volumes.get(str(step_idx), {})
    if not isinstance(step_entry, dict):
        return 0.0

    return float(step_entry.get(str(column_id), 0.0))


def normalize_distribution_values(step_idx, column_id, raw_values):
    raw_values = np.asarray(raw_values, dtype=np.float64)

    if raw_values.size == 0:
        return raw_values

    column_volume = get_column_volume(step_idx, column_id)
    depth = raw_values.size

    if column_volume <= 0 or depth <= 0:
        return np.zeros_like(raw_values, dtype=np.float64)

    slice_volume = column_volume / float(depth)

    if slice_volume <= 0:
        return np.zeros_like(raw_values, dtype=np.float64)

    return (raw_values / slice_volume) * 100.0


def compute_region_centers(voronoi_img):
    centers = {"x": [], "y": [], "label": []}

    for rid in np.unique(voronoi_img):
        if rid == 0:
            continue

        ys, xs = np.where(voronoi_img == rid)
        if len(xs) == 0:
            continue

        centers["x"].append(float(xs.mean()))
        centers["y"].append(float(ys.mean()))
        centers["label"].append(str(int(rid)))

    return centers


def compute_region_boundaries(voronoi_img):
    boundary = np.zeros_like(voronoi_img, dtype=np.uint8)

    boundary[:-1, :] |= (voronoi_img[:-1, :] != voronoi_img[1:, :])
    boundary[1:, :] |= (voronoi_img[:-1, :] != voronoi_img[1:, :])
    boundary[:, :-1] |= (voronoi_img[:, :-1] != voronoi_img[:, 1:])
    boundary[:, 1:] |= (voronoi_img[:, :-1] != voronoi_img[:, 1:])

    boundary[voronoi_img == 0] = 0
    return boundary.astype(np.float64)

def compute_selected_region_boundaries(voronoi_img, selected_ids, thickness=3):
    selected_ids = set(int(x) for x in selected_ids)

    mask = np.isin(voronoi_img, list(selected_ids))
    boundary = np.zeros_like(voronoi_img, dtype=np.uint8)

    boundary[:-1, :] |= (mask[:-1, :] != mask[1:, :])
    boundary[1:, :] |= (mask[:-1, :] != mask[1:, :])
    boundary[:, :-1] |= (mask[:, :-1] != mask[:, 1:])
    boundary[:, 1:] |= (mask[:, :-1] != mask[:, 1:])

    thick = boundary.copy()
    for _ in range(thickness - 1):
        expanded = thick.copy()
        expanded[:-1, :] |= thick[1:, :]
        expanded[1:, :] |= thick[:-1, :]
        expanded[:, :-1] |= thick[:, 1:]
        expanded[:, 1:] |= thick[:, :-1]
        thick = expanded

    return thick.astype(np.float64)


def compute_global_max(json_data):
    max_val = 0.0

    for step_dict in json_data.values():
        if not step_dict:
            continue
        step_max = max(float(v) for v in step_dict.values())
        max_val = max(max_val, step_max)

    return max(max_val, 1e-12)


def compute_distribution_limits(dist_data):
    max_x = 1.0
    max_y = 1

    for step_dict in dist_data.values():
        for vals in step_dict.values():
            if not vals:
                continue

            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                continue

            max_x = max(max_x, float(np.max(arr)))
            max_y = max(max_y, int(len(arr) - 1))

    return max_x, max_y


def available_steps_from_json(json_data):
    if not json_data:
        return np.array([0], dtype=np.int32)
    return np.array(sorted(int(k) for k in json_data.keys()), dtype=np.int32)


# =========================================================
# CLUSTER HELPERS
# =========================================================
def extract_cluster_features(series):
    series = np.asarray(series, dtype=np.float64)

    if series.size == 0:
        return {
            "mean": 0.0,
            "early_mean": 0.0,
            "mid_mean": 0.0,
            "late_mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "max": 0.0,
            "nonzero_ratio": 0.0,
            "early_nonzero_ratio": 0.0,
            "mid_nonzero_ratio": 0.0,
            "late_nonzero_ratio": 0.0,
        }

    n = len(series)
    i1 = max(1, n // 3)
    i2 = max(i1 + 1, 2 * n // 3)

    early = series[:i1]
    mid = series[i1:i2]
    late = series[i2:]

    def safe_mean(arr):
        return float(np.mean(arr)) if arr.size else 0.0

    def safe_nonzero_ratio(arr):
        return float(np.mean(arr > 0)) if arr.size else 0.0

    s = float(np.std(series))

    return {
        "mean": float(np.mean(series)),
        "early_mean": safe_mean(early),
        "mid_mean": safe_mean(mid),
        "late_mean": safe_mean(late),
        "std": s,
        "skewness": float(stats.skew(series)) if s > 0 else 0.0,
        "kurtosis": float(stats.kurtosis(series)) if s > 0 else 0.0,
        "max": float(np.max(series)),
        "nonzero_ratio": float(np.mean(series > 0)),
        "early_nonzero_ratio": safe_nonzero_ratio(early),
        "mid_nonzero_ratio": safe_nonzero_ratio(mid),
        "late_nonzero_ratio": safe_nonzero_ratio(late),
    }


def build_all_region_time_series(json_data, region_ids=None, step_range=None):
    steps = sorted(json_data.keys(), key=lambda k: int(k))

    if step_range is not None:
        start, end = step_range
        steps = [s for s in steps if start <= int(s) <= end]

    if region_ids is None:
        all_region_ids = set()
        for step_key in steps:
            all_region_ids.update(json_data.get(step_key, {}).keys())
        region_ids = sorted(int(rid) for rid in all_region_ids if int(rid) != 0)
    else:
        region_ids = sorted(int(rid) for rid in region_ids if int(rid) != 0)

    counts = {}
    for rid in region_ids:
        rid_str = str(rid)
        counts[rid] = np.array(
            [float(json_data.get(step, {}).get(rid_str, 0.0)) for step in steps],
            dtype=np.float64,
        )

    return counts


def sanitize_clustering_settings(linkage, metric):
    if linkage == "ward":
        return "ward", "euclidean"
    return linkage, metric


def compute_clustering(
    json_data,
    n_clusters,
    linkage,
    metric,
    selected_features,
    include_zero_columns,
    region_ids=None,
    step_range=None,
):
    counts = build_all_region_time_series(
        json_data,
        region_ids=region_ids,
        step_range=step_range,
    )

    feature_stats = {}
    for rid, series in counts.items():
        feature_stats[rid] = extract_cluster_features(series)

    if not feature_stats:
        return {}, pd.DataFrame(), counts, 0

    feature_df = pd.DataFrame(feature_stats).T
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if not include_zero_columns and "nonzero_ratio" in feature_df.columns:
        feature_df = feature_df[feature_df["nonzero_ratio"] > 0]

    if feature_df.empty:
        return {}, feature_df, counts, 0

    selected_features = [f for f in selected_features if f in feature_df.columns]
    if not selected_features:
        raise ValueError("Please select at least one valid clustering feature.")

    X = feature_df[selected_features].copy()
    if X.empty:
        return {}, feature_df, counts, 0

    if len(X) < 2:
        feature_df["cluster"] = 0
        cluster_map = {int(idx): 0 for idx in feature_df.index}
        return cluster_map, feature_df, counts, 1

    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    actual_n_clusters = min(int(n_clusters), len(X_scaled))
    actual_n_clusters = max(actual_n_clusters, 1)

    if actual_n_clusters == 1:
        feature_df["cluster"] = 0
        cluster_map = {int(idx): 0 for idx in feature_df.index}
        return cluster_map, feature_df, counts, 1

    linkage, metric = sanitize_clustering_settings(linkage, metric)

    if linkage == "ward":
        clustering = AgglomerativeClustering(
            n_clusters=actual_n_clusters,
            linkage=linkage,
        )
    else:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=actual_n_clusters,
                linkage=linkage,
                metric=metric,
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=actual_n_clusters,
                linkage=linkage,
                affinity=metric,
            )

    labels = clustering.fit_predict(X_scaled)

    feature_df["cluster"] = labels
    cluster_map = {int(idx): int(lbl) for idx, lbl in feature_df["cluster"].to_dict().items()}

    return cluster_map, feature_df, counts, actual_n_clusters


def create_cluster_map(voronoi_img, cluster_map):
    cluster_img = np.zeros_like(voronoi_img, dtype=np.float64)

    for rid, cluster_id in cluster_map.items():
        cluster_img[voronoi_img == rid] = float(cluster_id + 1)

    return cluster_img


# =========================================================
# DATA STORE
# =========================================================
def empty_data_store():
    return {
        "paths": {},
        "basis_files": [],
        "voronoi_files": [],
        "json_raw": {},
        "json_norm": {},
        "json_dist": {},
        "centroid_data": {"centroid": [0, 0, 0]},
        "z_unit": DEFAULT_Z_UNIT,
        "n_steps": 0,
        "dist_global_xmax": 1.0,
        "dist_global_ymax": 1,
        "time_steps_raw": np.array([0], dtype=np.int32),
        "time_steps_norm": np.array([0], dtype=np.int32),
        "all_region_ids": [],
        "cluster_map_raw": {},
        "cluster_map_norm": {},
        "cluster_features_raw": pd.DataFrame(),
        "cluster_features_norm": pd.DataFrame(),
        "cluster_counts_raw": {},
        "cluster_counts_norm": {},
        "cluster_n_raw": 0,
        "cluster_n_norm": 0,
    }


def load_all_data(base_dir):
    paths = build_paths(base_dir)

    basis_files = get_sorted_tiff_files(paths["basis"])
    voronoi_files = get_sorted_tiff_files(paths["voronoi"])
    all_region_ids = collect_all_region_ids_from_voronoi_files(voronoi_files)

    json_raw = load_json(paths["raw_json"])
    json_norm = load_json(paths["norm_json"])
    json_dist = load_json(paths["dist_json"])
    centroid_data = load_pickle(paths["centroid_pkl"])
    z_unit = load_z_unit(paths["voxel_spacings_pkl"])
    voronoi_volumes = load_json(paths["voronoi_volumes_json"])

    if len(basis_files) != len(voronoi_files):
        raise ValueError("Number of basis images and voronoi images does not match.")

    dist_global_xmax, dist_global_ymax = compute_distribution_limits(json_dist)

    return {
        "paths": paths,
        "basis_files": basis_files,
        "voronoi_files": voronoi_files,
        "json_raw": json_raw,
        "json_norm": json_norm,
        "json_dist": json_dist,
        "centroid_data": centroid_data,
        "z_unit": z_unit,
        "voronoi_volumes": voronoi_volumes,
        "n_steps": len(basis_files),
        "dist_global_xmax": max(dist_global_xmax, 1e-12),
        "dist_global_ymax": max(dist_global_ymax, 1),
        "time_steps_raw": available_steps_from_json(json_raw),
        "time_steps_norm": available_steps_from_json(json_norm),
        "all_region_ids": all_region_ids,
        "cluster_map_raw": {},
        "cluster_map_norm": {},
        "cluster_features_raw": pd.DataFrame(),
        "cluster_features_norm": pd.DataFrame(),
        "cluster_counts_raw": {},
        "cluster_counts_norm": {},
        "cluster_n_raw": 0,
        "cluster_n_norm": 0,
    }


# =========================================================
# INITIAL STATE
# =========================================================
data_store = empty_data_store()
app_ready = False
is_reloading = False

placeholder = np.zeros((32, 32), dtype=np.float64)
H0, W0 = placeholder.shape

cache = {
    "step_idx": INIT_STEP,
    "mode": INIT_MODE,
    "view_mode": INIT_VIEW_MODE,
    "alpha": INIT_ALPHA,
    "basis_img": placeholder,
    "voronoi_img": np.zeros_like(placeholder, dtype=np.int32),
    "step_values": {},
    "selected_region": None,
}

centroid_z = 0.0


# =========================================================
# WIDGETS
# =========================================================
scale_mode_select = pn.widgets.Select(
    name="Color scale mode",
    options=["global", "per-step"],
    value="per-step",
)

base_dir_input = pn.widgets.TextInput(name="Input base folder", value=DEFAULT_BASE_DIR)
reload_button = pn.widgets.Button(name="Load input folder", button_type="primary")
reload_status = pn.pane.Markdown("**Status:** waiting for a valid input folder")

normalize_switch = pn.widgets.Switch(name="Normalize", value=True)

alpha_slider = pn.widgets.FloatSlider(
    name="Overlay alpha",
    start=0.0,
    end=1.0,
    step=0.05,
    value=INIT_ALPHA,
)

step_slider = pn.widgets.IntSlider(
    name="Time step",
    start=0,
    end=1,
    value=0,
    disabled=True,
)

time_range_start_input = pn.widgets.IntInput(
    name="Start time step",
    value=0,
    start=0,
    end=0,
    disabled=True,
)

time_range_end_input = pn.widgets.IntInput(
    name="End time step",
    value=0,
    start=0,
    end=0,
    disabled=True,
)

apply_time_range_button = pn.widgets.Button(
    name="Apply time range",
    button_type="primary",
)

colormap_select = pn.widgets.Select(
    name="Colormap",
    options=list(PALETTES.keys()),
    value=INIT_CMAP,
)

border_color_picker = pn.widgets.ColorPicker(
    name="Border color",
    value="#00AEEF",
)

show_labels_toggle = pn.widgets.Checkbox(name="Show column numbers", value=False)
show_boundaries_toggle = pn.widgets.Checkbox(name="Show column boundaries", value=True)
show_colorbar_toggle = pn.widgets.Checkbox(name="Show scale bar", value=True)

show_timeseries_toggle = pn.widgets.Checkbox(name="Show column value over time", value=True)
show_distribution_toggle = pn.widgets.Checkbox(name="Show distribution at selected time step", value=True)
show_current_step_line_toggle = pn.widgets.Checkbox(name="Show current time step line", value=True)
flip_distribution_y_toggle = pn.widgets.Checkbox(name="Flip y-axis in distribution", value=False)

use_physical_z_axis_toggle = pn.widgets.Checkbox(
    name="Use physical z-axis",
    value=INIT_USE_PHYSICAL_Z_AXIS,
)

n_clusters_slider = pn.widgets.IntSlider(
    name="Number of clusters",
    value=INIT_N_CLUSTERS,
    start=2,
    end=min(7, len(CLUSTER_COLORS) - 1),
    step=1,
)

cluster_linkage_select = pn.widgets.Select(
    name="Clustering method",
    options=CLUSTER_LINKAGE_OPTIONS,
    value=INIT_CLUSTER_LINKAGE,
)

cluster_metric_select = pn.widgets.Select(
    name="Metric",
    options=CLUSTER_METRIC_OPTIONS,
    value=INIT_CLUSTER_METRIC,
)

include_zero_columns_toggle = pn.widgets.Switch(
    name="Include zero columns",
    value=INIT_INCLUDE_ZERO_COLUMNS,
)

show_mean_pattern_toggle = pn.widgets.Switch(
    name="Show mean pattern",
    value=INIT_SHOW_MEAN_PATTERN,
)

cluster_patterns_position_select = pn.widgets.Select(
    name="Clustered patterns position",
    options=[
        "below column-specific measurements",
        "right of compare columns",
    ],
    value="below column-specific measurements",
)

cluster_patterns_plots_per_row_select = pn.widgets.Select(
    name="Cluster plots per row",
    options=["1", "2"],
    value="2",
)

cluster_features_selector = pn.widgets.CheckBoxGroup(
    name="Features",
    value=INIT_CLUSTER_FEATURES,
    options=ALL_CLUSTER_FEATURES,
)

select_all_features_button = pn.widgets.Button(
    name="Select all",
    button_type="default",
    width=140,
)

deselect_all_features_button = pn.widgets.Button(
    name="Deselect all",
    button_type="default",
    width=150,
)

recompute_clusters_button = pn.widgets.Button(
    name="Recompute clustering",
    button_type="primary",
)

status_md = pn.pane.Markdown("**Selected column:** —")

view_tabs = pn.Tabs(
    ("Heatmap", pn.Spacer(height=1)),
    ("Clustering", pn.Spacer(height=1)),
    active=0,
    tabs_location="above",
    sizing_mode="stretch_width",
)

compare_column_ids_input = pn.widgets.TextInput(
    name="Column IDs",
    placeholder="e.g. 12, 18, 44",
)

compare_mode_selector = pn.widgets.CheckBoxGroup(
    name="Compare content",
    options=[
        "Show measurements",
        "Show cluster labels",
    ],
    value=["Show measurements"],
)

update_compare_button = pn.widgets.Button(
    name="Update comparison",
    button_type="primary",
)


# =========================================================
# DATA SOURCES
# =========================================================
basis_source = ColumnDataSource(data=dict(image=[placeholder], x=[0], y=[0], dw=[W0], dh=[H0]))
heatmap_source = ColumnDataSource(data=dict(image=[placeholder], x=[0], y=[0], dw=[W0], dh=[H0]))
cluster_source = ColumnDataSource(data=dict(image=[placeholder], x=[0], y=[0], dw=[W0], dh=[H0]))
boundary_source = ColumnDataSource(data=dict(image=[placeholder], x=[0], y=[0], dw=[W0], dh=[H0]))
label_source = ColumnDataSource(data=dict(x=[], y=[], label=[]))

selected_boundary_source = ColumnDataSource(data=dict(image=[placeholder], x=[0], y=[0], dw=[W0], dh=[H0]))

timeseries_source = ColumnDataSource(
    data=dict(
        x=np.array([0], dtype=np.int32),
        y=np.array([0.0], dtype=np.float64),
    )
)

distribution_source = ColumnDataSource(
    data=dict(
        x=np.array([], dtype=np.float64),
        y=np.array([], dtype=np.float64),
    )
)


# =========================================================
# FIGURES
# =========================================================
heatmap_color_mapper = LinearColorMapper(
    palette=PALETTES[INIT_CMAP],
    low=0.0,
    high=1.0,
    nan_color="#00000000",
)

cluster_color_mapper = LinearColorMapper(
    palette=CLUSTER_COLORS[1: INIT_N_CLUSTERS + 1],
    low=0.5,
    high=float(INIT_N_CLUSTERS) + 0.5,
    low_color="#FFFFFF",
    nan_color="#00000000",
)

basis_color_mapper = LinearColorMapper(
    palette=["#000000", "#FFFFFF"],
    low=0.0,
    high=1.0,
)

boundary_color_mapper = LinearColorMapper(
    palette=["#00000000", "#00AEEF"],
    low=0.0,
    high=1.0,
)

selected_boundary_color_mapper = LinearColorMapper(
    palette=["#00000000", "#FF2D2D"],
    low=0.0,
    high=1.0,
)

main_fig = figure(
    title="Heatmap Overlay",
    width=MAIN_FIG_WIDTH,
    height=MAIN_FIG_HEIGHT,
    sizing_mode="fixed",
    x_range=Range1d(0, W0),
    y_range=Range1d(H0, 0),
    tools="tap,box_zoom,reset,save",
    active_drag=None,
    active_scroll=None,
    match_aspect=True,
    output_backend="canvas",
)

basis_renderer = main_fig.image(
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    color_mapper=basis_color_mapper,
    source=basis_source,
)

heatmap_renderer = main_fig.image(
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    color_mapper=heatmap_color_mapper,
    source=heatmap_source,
    global_alpha=INIT_ALPHA,
    visible=True,
)

cluster_renderer = main_fig.image(
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    color_mapper=cluster_color_mapper,
    source=cluster_source,
    global_alpha=INIT_ALPHA,
    visible=False,
)

boundary_renderer = main_fig.image(
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    color_mapper=boundary_color_mapper,
    source=boundary_source,
)

selected_boundary_renderer = main_fig.image(
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    color_mapper=selected_boundary_color_mapper,
    source=selected_boundary_source,
)

labels_renderer = LabelSet(
    x="x",
    y="y",
    text="label",
    source=label_source,
    text_font_size="13pt",
    text_color="white",
    background_fill_alpha=0.0,
)
main_fig.add_layout(labels_renderer)

heatmap_color_bar = ColorBar(
    color_mapper=heatmap_color_mapper,
    label_standoff=8,
    visible=True,
    width=12,
    padding=6,
    title="Overlap",
    title_text_font_size="10pt",
    title_text_font_style="normal",
    major_label_text_font_size="9pt",
    ticker=BasicTicker(desired_num_ticks=5),
)

cluster_color_bar = ColorBar(
    color_mapper=cluster_color_mapper,
    visible=False,
    width=18,
    padding=4,
    label_standoff=6,
    title="",
    ticker=FixedTicker(ticks=list(range(1, INIT_N_CLUSTERS + 1))),
    major_label_overrides={i: f"C{i}" for i in range(1, INIT_N_CLUSTERS + 1)},
    major_label_policy=AllLabels(),
    major_label_text_font_size="8pt",
)

main_fig.add_layout(heatmap_color_bar, "right")
main_fig.add_layout(cluster_color_bar, "right")

main_fig.toolbar.logo = None
main_fig.xaxis.visible = False
main_fig.yaxis.visible = False
main_fig.xgrid.visible = False
main_fig.ygrid.visible = False
main_fig.outline_line_color = "#CCCCCC"
main_fig.background_fill_color = "white"
main_fig.border_fill_color = "white"


timeseries_fig = figure(
    title="Column value over time",
    width=DETAIL_FIG_WIDTH,
    height=DETAIL_FIG_HEIGHT,
    sizing_mode="fixed",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom",
    toolbar_location="right",
    output_backend="canvas",
    x_range=Range1d(0, 1),
    y_range=Range1d(0, 1),
)

timeseries_fig.line("x", "y", source=timeseries_source, line_width=2, color="#1f77b4")
ts_pts = timeseries_fig.scatter("x", "y", source=timeseries_source, size=6, color="#1f77b4", marker="circle")
timeseries_fig.add_tools(
    HoverTool(
        renderers=[ts_pts],
        tooltips=[("Time step", "@x"), ("Value", "@y{0.000000}")],
        mode="vline",
    )
)

timeseries_step_line = Span(
    location=0,
    dimension="height",
    line_color="#444444",
    line_width=2,
    line_dash="dashed",
    visible=True,
)
timeseries_fig.add_layout(timeseries_step_line)

timeseries_fig.toolbar.logo = None
timeseries_fig.xaxis.axis_label = "Time step"
timeseries_fig.yaxis.axis_label = "Value"
timeseries_fig.outline_line_color = "#CCCCCC"
timeseries_fig.background_fill_color = "white"
timeseries_fig.border_fill_color = "white"
timeseries_fig.grid.grid_line_alpha = 0.25


distribution_fig = figure(
    title="Distribution at selected time step",
    width=DETAIL_FIG_WIDTH,
    height=DETAIL_FIG_HEIGHT,
    sizing_mode="fixed",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom",
    toolbar_location="right",
    output_backend="canvas",
    x_range=Range1d(0, 1),
    y_range=Range1d(0, 1),
)

distribution_fig.line("x", "y", source=distribution_source, line_width=2, color="#ff7f0e")
dist_pts = distribution_fig.scatter("x", "y", source=distribution_source, size=5, color="#ff7f0e", marker="circle")
distribution_fig.add_tools(
    HoverTool(
        renderers=[dist_pts],
        tooltips=[("Z slice", "@y{0}"), ("Value", "@x{0.000000}")],
        mode="mouse",
    )
)

distribution_z_line = Span(
    location=0.0,
    dimension="width",
    line_color="red",
    line_width=2,
    line_dash="dashed",
)
distribution_fig.add_layout(distribution_z_line)

distribution_fig.toolbar.logo = None
distribution_fig.xaxis.axis_label = "Overlap"
distribution_fig.yaxis.axis_label = "Z slice"
distribution_fig.outline_line_color = "#CCCCCC"
distribution_fig.background_fill_color = "white"
distribution_fig.border_fill_color = "white"
distribution_fig.grid.grid_line_alpha = 0.25


compare_timeseries_fig = figure(
    title="Compare columns over time",
    width=COMPARE_FIG_WIDTH,
    height=COMPARE_FIG_HEIGHT,
    sizing_mode="fixed",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom",
    toolbar_location="right",
    output_backend="canvas",
    x_range=Range1d(0, 1),
    y_range=Range1d(0, 1),
)

compare_timeseries_fig.toolbar.logo = None
compare_timeseries_fig.xaxis.axis_label = "Time step"
compare_timeseries_fig.yaxis.axis_label = "Overlap"
compare_timeseries_fig.outline_line_color = "#CCCCCC"
compare_timeseries_fig.background_fill_color = "white"
compare_timeseries_fig.border_fill_color = "white"
compare_timeseries_fig.grid.grid_line_alpha = 0.25
compare_timeseries_fig.xaxis.visible = True
compare_timeseries_fig.yaxis.visible = True
compare_timeseries_fig.min_border_left = 70
compare_timeseries_fig.min_border_bottom = 60
compare_timeseries_fig.min_border_top = 20
compare_timeseries_fig.min_border_right = 20
compare_timeseries_fig.xaxis.axis_label_text_font_size = "14pt"
compare_timeseries_fig.yaxis.axis_label_text_font_size = "14pt"
compare_timeseries_fig.xaxis.major_label_text_font_size = "11pt"
compare_timeseries_fig.yaxis.major_label_text_font_size = "11pt"


compare_distribution_fig = figure(
    title="Compare distributions at current time step",
    width=COMPARE_FIG_WIDTH,
    height=COMPARE_FIG_HEIGHT,
    sizing_mode="fixed",
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom",
    toolbar_location="right",
    output_backend="canvas",
    x_range=Range1d(0, 1),
    y_range=Range1d(0, 1),
)

compare_distribution_fig.toolbar.logo = None
compare_distribution_fig.xaxis.axis_label = "Overlap"
compare_distribution_fig.yaxis.axis_label = "Z slice"
compare_distribution_fig.outline_line_color = "#CCCCCC"
compare_distribution_fig.background_fill_color = "white"
compare_distribution_fig.border_fill_color = "white"
compare_distribution_fig.grid.grid_line_alpha = 0.25
compare_distribution_fig.xaxis.visible = True
compare_distribution_fig.yaxis.visible = True
compare_distribution_fig.min_border_left = 70
compare_distribution_fig.min_border_bottom = 60
compare_distribution_fig.min_border_top = 20
compare_distribution_fig.min_border_right = 20
compare_distribution_fig.xaxis.axis_label_text_font_size = "14pt"
compare_distribution_fig.yaxis.axis_label_text_font_size = "14pt"
compare_distribution_fig.xaxis.major_label_text_font_size = "11pt"
compare_distribution_fig.yaxis.major_label_text_font_size = "11pt"

# Dummy glyphs to avoid MISSING_RENDERERS warnings
dummy_ts = compare_timeseries_fig.line([0, 1], [0, 0], line_alpha=0.0, line_width=0.0)
dummy_ts.tags = ["persistent_renderer"]

dummy_dist = compare_distribution_fig.line([0, 1], [0, 0], line_alpha=0.0, line_width=0.0)
dummy_dist.tags = ["persistent_renderer"]


# =========================================================
# STATE HELPERS
# =========================================================
def current_mode():
    return "normalized" if normalize_switch.value else "raw"


def current_view_mode():
    return "Heatmap" if view_tabs.active == 0 else "Clustering"


def get_z_unit():
    return float(data_store.get("z_unit", DEFAULT_Z_UNIT))


def get_distribution_y_values(n_slices):
    y = np.arange(n_slices, dtype=np.float64)

    if use_physical_z_axis_toggle.value:
        y = y * get_z_unit()

    return y


def get_distribution_y_label():
    if use_physical_z_axis_toggle.value:
        return "Depth [µm]"
    return "Depth [vx]"

def get_distribution_x_label():
    if current_mode() == "normalized":
        return "Overlap [%]"
    return "Overlap [vx]"

def get_timeseries_y_label():
    if current_mode() == "normalized":
        return "Overlap [%]"
    return "Overlap [vx]"


def get_active_time_range():
    if data_store["n_steps"] == 0:
        return 0, 0

    start = int(time_range_start_input.value)
    end = int(time_range_end_input.value)
    max_step = data_store["n_steps"] - 1

    start = max(0, min(start, max_step))
    end = max(0, min(end, max_step))

    if start > end:
        start, end = end, start

    return start, end


def filter_steps_by_active_range(steps):
    start, end = get_active_time_range()

    steps = np.asarray(steps, dtype=np.int32)
    filtered = steps[(steps >= start) & (steps <= end)]

    if filtered.size == 0:
        return np.array([start], dtype=np.int32)

    return filtered


def compute_global_max_in_active_range(json_data):
    start, end = get_active_time_range()
    max_val = 0.0

    for step_key, step_dict in json_data.items():
        step = int(step_key)

        if step < start or step > end:
            continue
        if not step_dict:
            continue

        step_max = max(float(v) for v in step_dict.values())
        max_val = max(max_val, step_max)

    return max(max_val, 1e-12)


def current_global_max(mode):
    if mode == "normalized":
        return compute_global_max_in_active_range(data_store["json_norm"])
    return compute_global_max_in_active_range(data_store["json_raw"])


def current_time_steps(mode):
    if mode == "normalized":
        return filter_steps_by_active_range(data_store["time_steps_norm"])
    return filter_steps_by_active_range(data_store["time_steps_raw"])


def get_active_cluster_state(mode):
    if mode == "normalized":
        return (
            data_store["cluster_map_norm"],
            data_store["cluster_counts_norm"],
            int(data_store["cluster_n_norm"]),
        )

    return (
        data_store["cluster_map_raw"],
        data_store["cluster_counts_raw"],
        int(data_store["cluster_n_raw"]),
    )


# =========================================================
# PLOT AXIS HELPERS
# =========================================================
def update_timeseries_axes(mode):
    steps = current_time_steps(mode)
    xmin = int(steps.min()) if steps.size else 0
    xmax = int(steps.max()) if steps.size else 0

    timeseries_fig.x_range.start = xmin
    timeseries_fig.x_range.end = max(xmax, xmin + 1)
    timeseries_fig.y_range.start = 0

    if mode == "normalized":
        timeseries_fig.y_range.end = current_global_max(mode) * 100.0 * 1.05
    else:
        timeseries_fig.y_range.end = current_global_max(mode) * 1.05

    timeseries_fig.yaxis.axis_label = get_timeseries_y_label()


def set_distribution_y_axis():
    ymax = max(int(data_store["dist_global_ymax"]), 1)

    if use_physical_z_axis_toggle.value:
        ymax *= get_z_unit()

    if flip_distribution_y_toggle.value:
        distribution_fig.y_range.start = ymax
        distribution_fig.y_range.end = 0
    else:
        distribution_fig.y_range.start = 0
        distribution_fig.y_range.end = ymax

    distribution_fig.yaxis.axis_label = get_distribution_y_label()


def update_heatmap_colorbar(mode, high_value):
    heatmap_color_bar.title = "Normalized overlap" if mode == "normalized" else "Overlap"
    heatmap_color_bar.ticker = BasicTicker(desired_num_ticks=5)

    if mode == "normalized":
        if high_value <= 1.0:
            heatmap_color_bar.formatter = PrintfTickFormatter(format="%.2f")
        else:
            heatmap_color_bar.formatter = PrintfTickFormatter(format="%.1f")
    else:
        if high_value >= 10000:
            heatmap_color_bar.formatter = NumeralTickFormatter(format="0.00a")
        elif high_value >= 100:
            heatmap_color_bar.formatter = PrintfTickFormatter(format="%.0f")
        elif high_value >= 1:
            heatmap_color_bar.formatter = PrintfTickFormatter(format="%.2f")
        else:
            heatmap_color_bar.formatter = PrintfTickFormatter(format="%.3f")


def update_cluster_colorbar(n_clusters):
    n_clusters = max(int(n_clusters), 1)

    cluster_color_mapper.palette = CLUSTER_COLORS[1: n_clusters + 1]
    cluster_color_mapper.low = 0.5
    cluster_color_mapper.high = float(n_clusters) + 0.5
    cluster_color_mapper.low_color = "#FFFFFF"

    ticks = list(range(1, n_clusters + 1))
    cluster_color_bar.ticker = FixedTicker(ticks=ticks)
    cluster_color_bar.major_label_overrides = {i: f"C{i}" for i in ticks}
    cluster_color_bar.major_label_policy = AllLabels()
    cluster_color_bar.major_label_text_font_size = "8pt"


def update_selected_compare_boundaries():
    selected_ids = parse_compare_column_ids()
    voronoi_img = cache["voronoi_img"]

    if voronoi_img is None or getattr(voronoi_img, "size", 0) == 0:
        selected_boundary_source.data = dict(
            image=[np.zeros((1, 1), dtype=np.float64)],
            x=[0],
            y=[0],
            dw=[1],
            dh=[1],
        )
        selected_boundary_renderer.visible = False
        return

    if len(selected_ids) == 0:
        empty = np.zeros_like(voronoi_img, dtype=np.float64)
        h, w = empty.shape
        selected_boundary_source.data = dict(
            image=[empty],
            x=[0],
            y=[0],
            dw=[w],
            dh=[h],
        )
        selected_boundary_renderer.visible = False
        return

    selected_boundary = compute_selected_region_boundaries(
        voronoi_img,
        selected_ids,
        thickness=4,
    )
    h, w = selected_boundary.shape

    selected_boundary_source.data = dict(
        image=[selected_boundary],
        x=[0],
        y=[0],
        dw=[w],
        dh=[h],
    )
    selected_boundary_renderer.visible = True

# =========================================================
# RESET / CLEANUP HELPERS
# =========================================================
def reset_detail_plots(mode):
    ts_x = current_time_steps(mode)

    timeseries_source.data = dict(
        x=ts_x,
        y=np.zeros(len(ts_x), dtype=np.float64),
    )

    distribution_source.data = dict(
        x=np.array([], dtype=np.float64),
        y=np.array([], dtype=np.float64),
    )

    update_timeseries_axes(mode)
    timeseries_step_line.location = step_slider.value
    timeseries_step_line.visible = show_current_step_line_toggle.value

    distribution_fig.x_range.start = 0
    distribution_fig.x_range.end = max(float(data_store["dist_global_xmax"]), 1.0)
    distribution_fig.xaxis.axis_label = get_distribution_x_label()
    set_distribution_y_axis()

    timeseries_fig.title.text = f"Column value over time ({mode})"
    distribution_fig.title.text = "Distribution at selected time step"


def clear_cluster_patterns():
    msg = [pn.pane.Markdown("Clustered column pattern plots appear here in the **Clustering** tab.")]
    cluster_patterns_container_bottom.objects = msg
    cluster_patterns_container_right.objects = msg
    cluster_patterns_panel_bottom.height = EMPTY_PANEL_HEIGHT
    cluster_patterns_panel_right.height = EMPTY_PANEL_HEIGHT


def clear_compare_figure(fig, title, x_label, y_label):
    keep_renderers = []

    for r in fig.renderers:
        if not hasattr(r, "glyph"):
            keep_renderers.append(r)
        elif "persistent_renderer" in getattr(r, "tags", []):
            keep_renderers.append(r)

    fig.renderers = keep_renderers

    for side_name in ["right", "center", "above", "below", "left"]:
        side = getattr(fig, side_name)
        legends_to_remove = [item for item in side if item.__class__.__name__ == "Legend"]
        for legend in legends_to_remove:
            side.remove(legend)

    fig.title.text = title
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label


# =========================================================
# CLUSTER LAYOUT HELPERS
# =========================================================
def get_cluster_pattern_plot_size():
    position = cluster_patterns_position_select.value
    per_row = cluster_patterns_plots_per_row_select.value

    if position == "right of compare columns":
        if per_row == "1":
            return 1040, 260
        return 500, 240

    if per_row == "1":
        return 1040, 260
    return 500, 280


def compute_cluster_panel_height(n_clusters):
    plots_per_row = int(cluster_patterns_plots_per_row_select.value)
    _, plot_height = get_cluster_pattern_plot_size()

    n_rows = int(np.ceil(n_clusters / plots_per_row))

    title_block = 60
    inner_padding = 24

    total_height = title_block + inner_padding + n_rows * plot_height + max(0, n_rows - 1) * ROW_GAP + 30
    return max(total_height, EMPTY_PANEL_HEIGHT)


def build_cluster_pattern_figure(cluster_id, cols_in_cluster, counts, steps, ymax, selected_region, show_mean_pattern):
    plot_width, plot_height = get_cluster_pattern_plot_size()

    fig = figure(
        title=f"Cluster {cluster_id + 1} — {len(cols_in_cluster)} columns",
        width=plot_width,
        height=plot_height,
        sizing_mode="fixed",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        toolbar_location="right",
        output_backend="canvas",
        x_range=Range1d(int(steps.min()) if steps.size else 0, max(int(steps.max()) if steps.size else 1, 1)),
        y_range=Range1d(0, ymax),
    )

    fig.xaxis.axis_label = "Time step"
    fig.yaxis.axis_label = "Overlap count"
    fig.outline_line_color = "#CCCCCC"
    fig.toolbar.logo = None
    fig.grid.grid_line_alpha = 0.25
    fig.grid.grid_line_dash = "dotted"
    fig.background_fill_color = "white"
    fig.border_fill_color = "white"

    line_color = CLUSTER_COLORS[min(cluster_id + 1, len(CLUSTER_COLORS) - 1)]

    series_stack = []
    for rid in cols_in_cluster:
        if rid not in counts:
            continue

        series = np.asarray(counts[rid], dtype=np.float64)
        if series.size == 0:
            continue

        series_stack.append(series)

        if selected_region is not None and int(rid) == int(selected_region):
            fig.line(
                x=steps,
                y=series,
                line_width=3.5,
                alpha=1.0,
                color="#D62728",
                legend_label=f"Selected column {rid}",
            )
        else:
            fig.line(
                x=steps,
                y=series,
                line_width=1.5,
                alpha=0.35,
                color=line_color,
            )

    if show_mean_pattern and series_stack:
        mean_curve = np.vstack(series_stack).mean(axis=0)
        fig.line(
            x=steps,
            y=mean_curve,
            line_width=3.0,
            color="black",
            alpha=0.95,
            legend_label="Mean pattern",
        )

    if fig.legend:
        fig.legend.location = "top_left"
        fig.legend.background_fill_alpha = 0.2

    return fig


def build_cluster_pattern_figures(counts, cluster_map, n_clusters, mode, selected_region, show_mean_pattern):
    if not cluster_map or n_clusters <= 0:
        return [pn.pane.Markdown("No clustering result available.")]

    steps = current_time_steps(mode)
    ymax = current_global_max(mode) * 1.05
    plots_per_row = int(cluster_patterns_plots_per_row_select.value)
    plot_width, plot_height = get_cluster_pattern_plot_size()

    plot_panes = []

    for cluster_id in range(n_clusters):
        cols_in_cluster = [rid for rid, lbl in cluster_map.items() if lbl == cluster_id]

        fig = build_cluster_pattern_figure(
            cluster_id=cluster_id,
            cols_in_cluster=cols_in_cluster,
            counts=counts,
            steps=steps,
            ymax=ymax,
            selected_region=selected_region,
            show_mean_pattern=show_mean_pattern,
        )

        plot_panes.append(
            pn.pane.Bokeh(
                fig,
                width=plot_width,
                height=plot_height,
                sizing_mode="fixed",
            )
        )

    rows = []

    if plots_per_row == 1:
        for pane in plot_panes:
            rows.append(
                pn.Row(
                    pane,
                    width=plot_width,
                    height=plot_height,
                    sizing_mode="fixed",
                )
            )
    else:
        row_width = plot_width * 2 + ROW_GAP

        for i in range(0, len(plot_panes), 2):
            row_items = plot_panes[i:i + 2]

            if len(row_items) == 2:
                rows.append(
                    pn.Row(
                        row_items[0],
                        pn.Spacer(width=ROW_GAP),
                        row_items[1],
                        width=row_width,
                        height=plot_height,
                        sizing_mode="fixed",
                    )
                )
            else:
                rows.append(
                    pn.Row(
                        row_items[0],
                        width=plot_width,
                        height=plot_height,
                        sizing_mode="fixed",
                    )
                )

    return rows


# =========================================================
# CORE UPDATE LOGIC
# =========================================================
def recompute_clustering_for_current_settings():
    if data_store["n_steps"] == 0:
        return

    selected_features = list(cluster_features_selector.value)
    if not selected_features:
        raise ValueError("Please select at least one clustering feature.")

    n_clusters = int(n_clusters_slider.value)
    linkage = str(cluster_linkage_select.value)
    metric = str(cluster_metric_select.value)
    include_zero_columns = bool(include_zero_columns_toggle.value)
    step_range = get_active_time_range()

    cluster_map_raw, cluster_features_raw, cluster_counts_raw, cluster_n_raw = compute_clustering(
        data_store["json_raw"],
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
        selected_features=selected_features,
        include_zero_columns=include_zero_columns,
        region_ids=data_store["all_region_ids"],
        step_range=step_range,
    )

    cluster_map_norm, cluster_features_norm, cluster_counts_norm, cluster_n_norm = compute_clustering(
        data_store["json_norm"],
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
        selected_features=selected_features,
        include_zero_columns=include_zero_columns,
        region_ids=data_store["all_region_ids"],
        step_range=step_range,
    )

    data_store["cluster_map_raw"] = cluster_map_raw
    data_store["cluster_map_norm"] = cluster_map_norm
    data_store["cluster_features_raw"] = cluster_features_raw
    data_store["cluster_features_norm"] = cluster_features_norm
    data_store["cluster_counts_raw"] = cluster_counts_raw
    data_store["cluster_counts_norm"] = cluster_counts_norm
    data_store["cluster_n_raw"] = cluster_n_raw
    data_store["cluster_n_norm"] = cluster_n_norm


def update_heatmap_overlay(step_idx, mode, alpha):
    step_key = str(step_idx)
    source_json = data_store["json_norm"] if mode == "normalized" else data_store["json_raw"]
    step_values = source_json.get(step_key, {})

    heatmap = create_heatmap(cache["voronoi_img"], step_values)
    h, w = heatmap.shape

    heatmap_source.data = dict(image=[heatmap], x=[0], y=[0], dw=[w], dh=[h])

    heatmap_color_mapper.palette = PALETTES[colormap_select.value]
    heatmap_color_mapper.low = 0.0

    if scale_mode_select.value == "global":
        high_value = current_global_max(mode)
    else:
        step_max = max((float(v) for v in step_values.values()), default=1e-12)
        high_value = max(step_max, 1e-12)

    heatmap_color_mapper.high = high_value
    update_heatmap_colorbar(mode, high_value)

    heatmap_renderer.glyph.global_alpha = alpha
    cache["step_values"] = step_values


def update_cluster_overlay(mode, alpha):
    cluster_map, _, n_clusters = get_active_cluster_state(mode)
    cluster_img = create_cluster_map(cache["voronoi_img"], cluster_map)

    h, w = cluster_img.shape
    cluster_source.data = dict(image=[cluster_img], x=[0], y=[0], dw=[w], dh=[h])

    cluster_renderer.glyph.global_alpha = alpha
    update_cluster_colorbar(max(n_clusters, 1))


def update_main_view(step_idx, mode, view_mode, alpha):
    basis_img, voronoi_img = load_step_images(step_idx, data_store["basis_files"], data_store["voronoi_files"])

    basis_norm = normalize01(basis_img)
    boundaries = compute_region_boundaries(voronoi_img)
    h, w = basis_norm.shape

    basis_source.data = dict(image=[basis_norm], x=[0], y=[0], dw=[w], dh=[h])
    boundary_source.data = dict(image=[boundaries], x=[0], y=[0], dw=[w], dh=[h])
    label_source.data = compute_region_centers(voronoi_img)

    boundary_color_mapper.palette = ["#00000000", border_color_picker.value]

    main_fig.x_range.start = 0
    main_fig.x_range.end = w
    main_fig.y_range.start = h
    main_fig.y_range.end = 0

    cache["step_idx"] = step_idx
    cache["mode"] = mode
    cache["view_mode"] = view_mode
    cache["alpha"] = alpha
    cache["basis_img"] = basis_img
    cache["voronoi_img"] = voronoi_img

    update_selected_compare_boundaries()

    if view_mode == "Heatmap":
        update_heatmap_overlay(step_idx, mode, alpha)
        heatmap_renderer.visible = True
        cluster_renderer.visible = False
        heatmap_color_bar.visible = show_colorbar_toggle.value
        cluster_color_bar.visible = False
        main_fig.title.text = "Heatmap Overlay"
    else:
        update_cluster_overlay(mode, alpha)
        heatmap_renderer.visible = False
        cluster_renderer.visible = True
        heatmap_color_bar.visible = False
        cluster_color_bar.visible = show_colorbar_toggle.value
        main_fig.title.text = "Cluster Map"


def update_timeseries(column_id, mode):
    source_json = data_store["json_norm"] if mode == "normalized" else data_store["json_raw"]
    ts_x, ts_y = build_time_series(column_id, source_json, step_range=get_active_time_range())

    if ts_x.size == 0:
        ts_x = np.array([0], dtype=np.int32)
        ts_y = np.array([0.0], dtype=np.float64)

    if mode == "normalized":
        ts_y = ts_y * 100.0

    timeseries_source.data = dict(x=ts_x, y=ts_y)
    update_timeseries_axes(mode)
    timeseries_step_line.location = cache["step_idx"]
    timeseries_step_line.visible = show_current_step_line_toggle.value
    timeseries_fig.title.text = f"Column {column_id} value over time ({mode})"


def update_distribution(column_id, step_idx):
    distribution_fig.x_range.start = 0
    set_distribution_y_axis()

    dist_vals = build_distribution(column_id, data_store["json_dist"], step_idx)

    if len(dist_vals) == 0:
        distribution_source.data = dict(
            x=np.array([], dtype=np.float64),
            y=np.array([], dtype=np.float64),
        )
        distribution_fig.title.text = f"Column {column_id} — no distribution data at time step {step_idx}"
        distribution_fig.xaxis.axis_label = get_distribution_x_label()
        distribution_z_line.location = centroid_z * get_z_unit() if use_physical_z_axis_toggle.value else centroid_z
        return

    if current_mode() == "normalized":
        x = normalize_distribution_values(step_idx, column_id, dist_vals)
    else:
        x = dist_vals.astype(np.float64)

    y = get_distribution_y_values(len(dist_vals))

    distribution_source.data = dict(x=x, y=y)
    distribution_fig.title.text = f"Column {column_id} distribution at time step {step_idx}"
    distribution_fig.xaxis.axis_label = get_distribution_x_label()

    distribution_fig.x_range.start = 0
    distribution_fig.x_range.end = max(float(np.max(x)), 1.0) * 1.05

    distribution_z_line.location = centroid_z * get_z_unit() if use_physical_z_axis_toggle.value else centroid_z


def update_selection_status(region_id, mode, view_mode):
    msg = f"**Selected column:** {region_id}"

    if view_mode == "Clustering":
        cluster_map, _, _ = get_active_cluster_state(mode)
        cluster_id = cluster_map.get(int(region_id), None)
        if cluster_id is not None:
            msg += f"  |  **Cluster:** {cluster_id + 1}"
        else:
            msg += "  |  **Cluster:** —"

    status_md.object = msg


def update_cluster_patterns(mode, view_mode):
    if view_mode != "Clustering":
        clear_cluster_patterns()
        return

    cluster_map, counts, n_clusters = get_active_cluster_state(mode)

    figs = build_cluster_pattern_figures(
        counts=counts,
        cluster_map=cluster_map,
        n_clusters=n_clusters,
        mode=mode,
        selected_region=cache["selected_region"],
        show_mean_pattern=show_mean_pattern_toggle.value,
    )

    dynamic_height = compute_cluster_panel_height(n_clusters)

    if cluster_patterns_position_select.value == "right of compare columns":
        cluster_patterns_container_right.objects = figs
        cluster_patterns_container_bottom.objects = []
        cluster_patterns_panel_right.height = dynamic_height
    else:
        cluster_patterns_container_bottom.objects = figs
        cluster_patterns_container_right.objects = []
        cluster_patterns_panel_bottom.height = dynamic_height


def update_compare_panel():
    mode = current_mode()
    selected_ids = parse_compare_column_ids()

    show_measurements = "Show measurements" in compare_mode_selector.value
    show_cluster_context = "Show cluster labels" in compare_mode_selector.value
    draw_compare_lines = show_measurements or show_cluster_context

    clear_compare_figure(
        compare_timeseries_fig,
        title="Compare columns over time",
        x_label="Time step",
        y_label=get_timeseries_y_label(),
    )
    clear_compare_figure(
        compare_distribution_fig,
        title="Compare distributions at current time step",
        x_label="Overlap",
        y_label=get_distribution_y_label(),
    )

    if not selected_ids:
        compare_timeseries_fig.x_range.start = 0
        compare_timeseries_fig.x_range.end = 1
        compare_timeseries_fig.y_range.start = 0
        compare_timeseries_fig.y_range.end = 1

        compare_distribution_fig.x_range.start = 0
        compare_distribution_fig.x_range.end = 1
        compare_distribution_fig.y_range.start = 0
        compare_distribution_fig.y_range.end = 1
        return

    source_json = data_store["json_norm"] if mode == "normalized" else data_store["json_raw"]
    cluster_map, _, _ = get_active_cluster_state(mode)

    colors = CLUSTER_COLORS[1:] + ["#111111", "#17becf", "#bcbd22"]

    steps = current_time_steps(mode)
    xmin = int(steps.min()) if steps.size else 0
    xmax = int(steps.max()) if steps.size else 1
    if mode == "normalized":
        ymax = current_global_max(mode) * 100.0 * 1.05
    else:
        ymax = current_global_max(mode) * 1.05

    compare_timeseries_fig.x_range.start = xmin
    compare_timeseries_fig.x_range.end = max(xmax, xmin + 1)
    compare_timeseries_fig.y_range.start = 0
    compare_timeseries_fig.y_range.end = ymax

    dist_xmax = 1.0
    dist_ymax = 1.0
    has_any_distribution = False

    for rid in selected_ids:
        dist_vals = build_distribution(rid, data_store["json_dist"], step_slider.value)

        if len(dist_vals) > 0:
            has_any_distribution = True

            if mode == "normalized":
                x_vals = normalize_distribution_values(step_slider.value, rid, dist_vals)
            else:
                x_vals = dist_vals.astype(np.float64)

            dist_xmax = max(dist_xmax, float(np.max(x_vals)))

            y_vals = get_distribution_y_values(len(dist_vals))
            if len(y_vals) > 0:
                dist_ymax = max(dist_ymax, float(np.max(y_vals)))

    if not has_any_distribution:
        if use_physical_z_axis_toggle.value:
            dist_ymax = max(float(data_store["dist_global_ymax"]) * get_z_unit(), 1.0)
        else:
            dist_ymax = max(float(data_store["dist_global_ymax"]), 1.0)

    compare_distribution_fig.x_range.start = 0
    compare_distribution_fig.x_range.end = max(dist_xmax * 1.05, 1.0)

    if flip_distribution_y_toggle.value:
        compare_distribution_fig.y_range.start = dist_ymax
        compare_distribution_fig.y_range.end = 0
    else:
        compare_distribution_fig.y_range.start = 0
        compare_distribution_fig.y_range.end = dist_ymax

    compare_distribution_fig.yaxis.axis_label = get_distribution_y_label()
    compare_distribution_fig.xaxis.axis_label = get_distribution_x_label()

    for i, rid in enumerate(selected_ids):
        color = colors[i % len(colors)]

        label = f"Column {rid}"
        if show_cluster_context:
            cid = cluster_map.get(rid, None)
            if cid is not None:
                label = f"Column {rid} (C{cid + 1})"

        if draw_compare_lines:
            ts_x, ts_y = build_time_series(rid, source_json, step_range=get_active_time_range())

            if mode == "normalized":
                ts_y = ts_y * 100.0

            if ts_x.size > 0:
                compare_timeseries_fig.line(
                    x=ts_x,
                    y=ts_y,
                    line_width=2.5,
                    color=color,
                    alpha=0.95,
                    legend_label=label,
                )

            dist_vals = build_distribution(rid, data_store["json_dist"], step_slider.value)

            if len(dist_vals) > 0:
                if mode == "normalized":
                    x = normalize_distribution_values(step_slider.value, rid, dist_vals)
                else:
                    x = dist_vals.astype(np.float64)

                y = get_distribution_y_values(len(dist_vals))
            else:
                if use_physical_z_axis_toggle.value:
                    max_y = max(float(data_store["dist_global_ymax"]) * get_z_unit(), 1.0)
                else:
                    max_y = max(float(data_store["dist_global_ymax"]), 1.0)

                n_points = max(int(data_store["dist_global_ymax"]) + 1, 2)
                y = np.linspace(0.0, max_y, n_points, dtype=np.float64)
                x = np.zeros_like(y, dtype=np.float64)

            compare_distribution_fig.line(
                x=x,
                y=y,
                line_width=2.5,
                color=color,
                alpha=0.95,
                legend_label=label,
            )

    if compare_timeseries_fig.legend:
        compare_timeseries_fig.legend.location = "top_right"
        compare_timeseries_fig.legend.background_fill_alpha = 0.65
        compare_timeseries_fig.legend.click_policy = "hide"
        compare_timeseries_fig.legend.label_text_font_size = "9pt"

    if compare_distribution_fig.legend:
        compare_distribution_fig.legend.location = "top_right"
        compare_distribution_fig.legend.background_fill_alpha = 0.65
        compare_distribution_fig.legend.click_policy = "hide"
        compare_distribution_fig.legend.label_text_font_size = "9pt"


def update_ui_visibility(view_mode):
    boundary_renderer.visible = show_boundaries_toggle.value
    labels_renderer.visible = show_labels_toggle.value

    timeseries_panel.visible = show_timeseries_toggle.value
    distribution_panel.visible = show_distribution_toggle.value

    show_cluster_patterns = (view_mode == "Clustering")
    show_right = (
        show_cluster_patterns
        and cluster_patterns_position_select.value == "right of compare columns"
    )
    show_bottom = (
        show_cluster_patterns
        and cluster_patterns_position_select.value == "below column-specific measurements"
    )

    cluster_patterns_panel_right.visible = show_right
    cluster_patterns_panel_bottom.visible = show_bottom

    timeseries_step_line.visible = show_current_step_line_toggle.value and show_timeseries_toggle.value

    heatmap_data_section.visible = (view_mode == "Heatmap")
    heatmap_appearance_section.visible = True
    cluster_section.visible = (view_mode == "Clustering")


def refresh():
    mode = current_mode()
    view_mode = current_view_mode()

    update_ui_visibility(view_mode)

    if not app_ready or data_store["n_steps"] == 0:
        reset_detail_plots(mode)
        clear_cluster_patterns()
        return

    update_main_view(
        step_idx=step_slider.value,
        mode=mode,
        view_mode=view_mode,
        alpha=alpha_slider.value,
    )

    if cache["selected_region"] is not None:
        rid = cache["selected_region"]
        update_timeseries(rid, mode)
        update_distribution(rid, step_slider.value)
        update_selection_status(rid, mode, view_mode)
    else:
        status_md.object = "**Selected column:** —"
        reset_detail_plots(mode)

    update_cluster_patterns(mode, view_mode)
    update_compare_panel()


# =========================================================
# CALLBACKS
# =========================================================
def on_widget_change(event):
    if is_reloading:
        return
    refresh()


def on_select_all_features(event=None):
    cluster_features_selector.value = list(ALL_CLUSTER_FEATURES)


def on_deselect_all_features(event=None):
    cluster_features_selector.value = []


def on_apply_time_range(event=None):
    if data_store["n_steps"] == 0:
        return

    start, end = get_active_time_range()

    time_range_start_input.value = start
    time_range_end_input.value = end

    step_slider.start = start
    step_slider.end = end

    if step_slider.value < start:
        step_slider.value = start
    elif step_slider.value > end:
        step_slider.value = end

    if app_ready:
        try:
            recompute_clustering_for_current_settings()
        except Exception as e:
            reload_status.object = f"**Status:** clustering failed — {e}"

    refresh()


def on_cluster_linkage_change(event=None):
    if cluster_linkage_select.value == "ward":
        cluster_metric_select.value = "euclidean"
        cluster_metric_select.disabled = True
    else:
        cluster_metric_select.disabled = False


def reload_data(event=None):
    global data_store, centroid_z, app_ready, is_reloading

    if is_reloading:
        return

    is_reloading = True
    try:
        new_base_dir = base_dir_input.value.strip()
        if not new_base_dir:
            raise ValueError("Please enter a base directory.")

        data_store = load_all_data(new_base_dir)
        app_ready = True
        cache["selected_region"] = None
        centroid_z = float(data_store["centroid_data"]["centroid"][2])

        max_step = max(data_store["n_steps"] - 1, 0)

        step_slider.disabled = False
        step_slider.start = 0
        step_slider.end = max_step
        step_slider.value = 0

        time_range_start_input.disabled = False
        time_range_end_input.disabled = False

        time_range_start_input.start = 0
        time_range_start_input.end = max_step
        time_range_start_input.value = 0

        time_range_end_input.start = 0
        time_range_end_input.end = max_step
        time_range_end_input.value = max_step

        distribution_z_line.location = centroid_z * get_z_unit() if use_physical_z_axis_toggle.value else centroid_z

        recompute_clustering_for_current_settings()

        reload_status.object = f"**Status:** loaded "
        refresh()

    except Exception as e:
        app_ready = False
        data_store = empty_data_store()
        cache["selected_region"] = None
        centroid_z = 0.0

        step_slider.disabled = True
        step_slider.start = 0
        step_slider.end = 1
        step_slider.value = 0

        time_range_start_input.disabled = True
        time_range_end_input.disabled = True

        time_range_start_input.start = 0
        time_range_start_input.end = 0
        time_range_start_input.value = 0

        time_range_end_input.start = 0
        time_range_end_input.end = 0
        time_range_end_input.value = 0

        reload_status.object = f"**Status:** reload failed — {e}"
        status_md.object = "**Selected column:** —"
        reset_detail_plots(current_mode())
        clear_cluster_patterns()

    finally:
        is_reloading = False


def on_recompute_clusters(event=None):
    if not app_ready or data_store["n_steps"] == 0:
        return

    try:
        recompute_clustering_for_current_settings()
        refresh()
    except Exception as e:
        reload_status.object = f"**Status:** clustering failed — {e}"


def on_tap(event):
    if data_store["n_steps"] == 0:
        return

    rid = get_region_id(event.x, event.y, cache["voronoi_img"])

    if rid is None:
        cache["selected_region"] = None
        status_md.object = "**Selected column:** outside image"
        reset_detail_plots(current_mode())
        return

    if rid == 0:
        cache["selected_region"] = None
        status_md.object = "**Selected column:** background"
        reset_detail_plots(current_mode())
        return

    cache["selected_region"] = rid
    update_timeseries(rid, current_mode())
    update_distribution(rid, cache["step_idx"])
    update_selection_status(rid, current_mode(), current_view_mode())
    update_cluster_patterns(current_mode(), current_view_mode())


def on_update_compare(event=None):
    refresh()


def on_mousewheel(event):
    if data_store["n_steps"] == 0:
        return

    delta = -1 if event.delta > 0 else 1
    new_value = int(np.clip(step_slider.value + delta, step_slider.start, step_slider.end))

    if new_value != step_slider.value:
        step_slider.value = new_value


main_fig.on_event(Tap, on_tap)
main_fig.on_event(MouseWheel, on_mousewheel)


# =========================================================
# WATCHERS
# =========================================================
step_slider.param.watch(on_widget_change, "value")
normalize_switch.param.watch(on_widget_change, "value")
alpha_slider.param.watch(on_widget_change, "value")
colormap_select.param.watch(on_widget_change, "value")
scale_mode_select.param.watch(on_widget_change, "value")
border_color_picker.param.watch(on_widget_change, "value")
show_labels_toggle.param.watch(on_widget_change, "value")
show_boundaries_toggle.param.watch(on_widget_change, "value")
show_colorbar_toggle.param.watch(on_widget_change, "value")
show_timeseries_toggle.param.watch(on_widget_change, "value")
show_distribution_toggle.param.watch(on_widget_change, "value")
show_current_step_line_toggle.param.watch(on_widget_change, "value")
flip_distribution_y_toggle.param.watch(on_widget_change, "value")
use_physical_z_axis_toggle.param.watch(on_widget_change, "value")
show_mean_pattern_toggle.param.watch(on_widget_change, "value")
view_tabs.param.watch(on_widget_change, "active")

compare_mode_selector.param.watch(on_widget_change, "value")
cluster_linkage_select.param.watch(on_cluster_linkage_change, "value")
cluster_patterns_position_select.param.watch(on_widget_change, "value")
cluster_patterns_plots_per_row_select.param.watch(on_widget_change, "value")

reload_button.on_click(reload_data)
apply_time_range_button.on_click(on_apply_time_range)
recompute_clusters_button.on_click(on_recompute_clusters)
select_all_features_button.on_click(on_select_all_features)
deselect_all_features_button.on_click(on_deselect_all_features)
update_compare_button.on_click(on_update_compare)

on_cluster_linkage_change()


# =========================================================
# STYLES
# =========================================================
SECTION_STYLE = {
    "background": "#FFFFFF",
    "border": "1px solid #D9D9D9",
    "border-radius": "10px",
    "padding": "14px",
    "margin-bottom": "14px",
}

STATUS_STYLE = {
    "background": "#FFFFFF",
    "border": "1px solid #D9D9D9",
    "border-radius": "8px",
    "padding": "10px",
    "font-size": "0.95rem",
    "line-height": "1.4",
    "overflow-wrap": "break-word",
    "color": "#222222",
}

PLOT_SECTION_STYLE = {
    "background": "#FFFFFF",
    "border": "1px solid #D9D9D9",
    "border-radius": "12px",
    "padding": "16px",
    "margin-bottom": "18px",
}

reload_status.styles = STATUS_STYLE
status_md.styles = STATUS_STYLE


# =========================================================
# LAYOUT
# =========================================================
input_section = pn.Column(
    pn.pane.Markdown("### Input"),
    base_dir_input,
    reload_button,
    reload_status,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

time_range_section = pn.Column(
    pn.pane.Markdown("### Time range"),
    time_range_start_input,
    time_range_end_input,
    apply_time_range_button,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

heatmap_data_section = pn.Column(
    pn.pane.Markdown("### Heatmap data"),
    normalize_switch,
    scale_mode_select,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

heatmap_appearance_section = pn.Column(
    pn.pane.Markdown("### Overlay appearance"),
    alpha_slider,
    colormap_select,
    border_color_picker,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

cluster_section = pn.Column(
    pn.pane.Markdown("### Clustering"),
    n_clusters_slider,
    cluster_linkage_select,
    cluster_metric_select,
    cluster_patterns_position_select,
    cluster_patterns_plots_per_row_select,
    pn.Column(
        include_zero_columns_toggle,
        show_mean_pattern_toggle,
        styles={"margin-top": "8px", "margin-bottom": "14px"},
        sizing_mode="stretch_width",
    ),
    pn.pane.Markdown("#### Features"),
    cluster_features_selector,
    pn.Row(
        select_all_features_button,
        deselect_all_features_button,
        width=300,
        height=36,
        sizing_mode="fixed",
    ),
    recompute_clusters_button,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

display_section = pn.Column(
    pn.pane.Markdown("### Display"),
    show_labels_toggle,
    show_boundaries_toggle,
    show_colorbar_toggle,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

detail_plots_section = pn.Column(
    pn.pane.Markdown("### Detail plots"),
    show_timeseries_toggle,
    show_distribution_toggle,
    show_current_step_line_toggle,
    flip_distribution_y_toggle,
    use_physical_z_axis_toggle,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

compare_section = pn.Column(
    pn.pane.Markdown("### Compare columns"),
    compare_column_ids_input,
    compare_mode_selector,
    update_compare_button,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

selection_section = pn.Column(
    pn.pane.Markdown("### Selection"),
    status_md,
    styles=SECTION_STYLE,
    sizing_mode="stretch_width",
)

sidebar_content = pn.Column(
    pn.pane.Markdown("# Controls"),
    input_section,
    time_range_section,
    heatmap_data_section,
    heatmap_appearance_section,
    cluster_section,
    compare_section,
    display_section,
    detail_plots_section,
    selection_section,
    sizing_mode="stretch_width",
    styles={
        "padding-left": "8px",
        "padding-right": "8px",
    },
)

timeseries_panel = pn.Column(
    pn.pane.Bokeh(timeseries_fig, sizing_mode="fixed"),
    visible=show_timeseries_toggle.value,
    width=DETAIL_FIG_WIDTH,
    height=DETAIL_FIG_HEIGHT,
    sizing_mode="fixed",
)

distribution_panel = pn.Column(
    pn.pane.Bokeh(distribution_fig, sizing_mode="fixed"),
    visible=show_distribution_toggle.value,
    width=DETAIL_FIG_WIDTH,
    height=DETAIL_FIG_HEIGHT,
    sizing_mode="fixed",
)

cluster_patterns_container_bottom = pn.Column(
    width=SECTION_WIDTH,
    height=EMPTY_PANEL_HEIGHT,
    sizing_mode="fixed",
    styles={"align-items": "flex-start"},
)

cluster_patterns_container_right = pn.Column(
    width=RIGHT_CLUSTER_PANEL_WIDTH,
    height=EMPTY_PANEL_HEIGHT,
    sizing_mode="fixed",
    styles={"align-items": "flex-start"},
)

cluster_patterns_panel_bottom = pn.Column(
    pn.pane.Markdown("## Clustered column patterns"),
    cluster_patterns_container_bottom,
    visible=False,
    styles={**PLOT_SECTION_STYLE, "align-items": "flex-start"},
    width=SECTION_WIDTH,
    height=EMPTY_PANEL_HEIGHT,
    sizing_mode="fixed",
)

cluster_patterns_panel_right = pn.Column(
    pn.pane.Markdown("## Clustered column patterns"),
    cluster_patterns_container_right,
    visible=False,
    styles={**PLOT_SECTION_STYLE, "align-items": "flex-start"},
    width=RIGHT_CLUSTER_PANEL_WIDTH,
    height=EMPTY_PANEL_HEIGHT,
    sizing_mode="fixed",
)

compare_panel = pn.Column(
    pn.pane.Markdown("## Compare columns"),
    pn.pane.Bokeh(compare_timeseries_fig, sizing_mode="fixed"),
    pn.Spacer(height=16),
    pn.pane.Bokeh(compare_distribution_fig, sizing_mode="fixed"),
    styles=PLOT_SECTION_STYLE,
    width=COMPARE_PANEL_WIDTH,
    height=TOP_SECTION_HEIGHT,
    sizing_mode="fixed",
)

main_map_section = pn.Column(
    pn.pane.Markdown("## Spatial overview"),
    view_tabs,
    pn.pane.Bokeh(main_fig, sizing_mode="fixed"),
    pn.Spacer(height=8),
    step_slider,
    styles=PLOT_SECTION_STYLE,
    width=SECTION_WIDTH,
    height=TOP_SECTION_HEIGHT,
    sizing_mode="fixed",
)

top_row_section = pn.Row(
    main_map_section,
    pn.Spacer(width=20),
    compare_panel,
    pn.Spacer(width=20),
    cluster_patterns_panel_right,
    width=MAIN_CONTENT_WIDTH,
    height=TOP_SECTION_HEIGHT,
    sizing_mode="fixed",
)

detail_plots_row = pn.Row(
    timeseries_panel,
    pn.Spacer(width=30),
    distribution_panel,
    width=DETAIL_FIG_WIDTH * 2 + 30,
    height=DETAIL_FIG_HEIGHT,
    sizing_mode="fixed",
)

column_detail_section = pn.Column(
    pn.pane.Markdown("## Column-specific measurements"),
    detail_plots_row,
    styles=PLOT_SECTION_STYLE,
    width=SECTION_WIDTH,
    height=DETAIL_FIG_HEIGHT + 90,
    sizing_mode="fixed",
)

main_content = pn.Column(
    top_row_section,
    pn.Spacer(height=14),
    column_detail_section,
    pn.Spacer(height=14),
    cluster_patterns_panel_bottom,
    width=MAIN_CONTENT_WIDTH,
    sizing_mode="fixed",
)

# =========================================================
# HELP / ABOUT
# =========================================================
help_button = pn.widgets.Button(
    name="❓ Help",
    button_type="light",
    width=95,
    height=36,
)

help_html = """
<div style="max-width: 980px; padding: 10px 6px 10px 6px; line-height: 1.6;">

<h1 style="margin-top: 0;">Column Explorer – Help</h1>

<p>
<b>Column Explorer</b> is an interactive application for exploring spatial overlap data
across Voronoi-based columns over time. It combines image overlays, temporal plots,
distribution views, clustering and column comparison in one interface.
</p>

<h2>What is this explorer for?</h2>
<p>
The explorer helps you inspect how values change across spatial regions and time steps.
It is useful for identifying local patterns, comparing columns, understanding temporal
behavior, and grouping columns with similar dynamics by clustering.
</p>

<h2>Main capabilities</h2>
<ul>
  <li><b>Heatmap overlay:</b> shows region values on top of the spatial image.</li>
  <li><b>Cluster map:</b> shows cluster assignments directly on the Voronoi regions.</li>
  <li><b>Column selection:</b> click a column in the spatial map to inspect it in detail.</li>
  <li><b>Time-series view:</b> shows how the selected column changes over time.</li>
  <li><b>Distribution view:</b> shows the z-distribution for the selected column at the current time step.</li>
  <li><b>Column comparison:</b> compare several columns side by side.</li>
  <li><b>Clustering:</b> group columns by temporal summary features.</li>
  <li><b>Clustered column patterns:</b> inspect temporal behavior of all columns inside each cluster.</li>
  <li><b>Time range filtering:</b> restrict analysis and clustering to a selected interval.</li>
</ul>

<h2>Input data</h2>
<p>
The application expects a base folder.
</p>

<h2>How to use the explorer</h2>
<ol>
  <li>Enter the input base folder and click <b>Load input folder</b>.</li>
  <li>Choose a time range and apply it if needed.</li>
  <li>Use the <b>Heatmap</b> tab to inspect spatial overlap values.</li>
  <li>Use the <b>Clustering</b> tab to inspect grouped column behavior.</li>
  <li>Click on a column in the map to open its detailed measurements.</li>
  <li>Use the compare section to inspect multiple columns together.</li>
</ol>

<h2>Controls explained</h2>

<h3>Input</h3>
<ul>
  <li><b>Input base folder:</b> root folder containing all required files.</li>
  <li><b>Load input folder:</b> loads all files and initializes the explorer.</li>
</ul>

<h3>Time range</h3>
<ul>
  <li><b>Start time step / End time step:</b> defines the active analysis interval.</li>
  <li><b>Apply time range:</b> updates plots, scaling, and clustering using only the selected interval.</li>
</ul>

<h3>Heatmap data</h3>
<ul>
  <li><b>Normalize:</b> switches between normalized and raw overlap values.</li>
  <li><b>Color scale mode:</b>
    <ul>
      <li><b>global</b> – same scale across the active time range</li>
      <li><b>per-step</b> – scale adapts to the current time step</li>
    </ul>
  </li>
</ul>

<h3>Overlay appearance</h3>
<ul>
  <li><b>Overlay alpha:</b> controls transparency of the overlay.</li>
  <li><b>Colormap:</b> changes the heatmap color palette.</li>
  <li><b>Border color:</b> changes Voronoi boundary color.</li>
</ul>

<h3>Clustering</h3>
<ul>
  <li><b>Number of clusters:</b> number of groups to compute.</li>
  <li><b>Clustering method:</b> hierarchical linkage strategy.</li>
  <li><b>Metric:</b> distance metric used for clustering.</li>
  <li><b>Include zero columns:</b> includes columns with zero activity.</li>
  <li><b>Show mean pattern:</b> overlays the mean time-series for each cluster.</li>
  <li><b>Clustered patterns position:</b> places cluster plots either below or to the right.</li>
  <li><b>Cluster plots per row:</b> choose 1 or 2 plots per row.</li>
  <li><b>Features:</b> select summary features used for clustering.</li>
  <li><b>Recompute clustering:</b> runs clustering again with current settings.</li>
</ul>

<h3>Display</h3>
<ul>
  <li><b>Show column numbers:</b> toggles region labels.</li>
  <li><b>Show column boundaries:</b> toggles Voronoi boundaries.</li>
  <li><b>Show scale bar:</b> toggles the colorbar.</li>
</ul>

<h3>Detail plots</h3>
<ul>
  <li><b>Show column value over time:</b> toggles selected-column time-series.</li>
  <li><b>Show distribution at selected time step:</b> toggles z-distribution plot.</li>
  <li><b>Show current time step line:</b> displays the active step marker in the time-series.</li>
  <li><b>Flip y-axis in distribution:</b> inverts the distribution y-axis.</li>
  <li><b>Use physical z-axis:</b> scales z using the physical z-unit.</li>
</ul>

<h3>Compare columns</h3>
<ul>
  <li><b>Column IDs:</b> enter comma-separated IDs, for example <code>12, 18, 44</code>.</li>
  <li><b>Show measurements:</b> compares selected columns by values.</li>
  <li><b>Show cluster labels:</b> adds cluster context in the legend.</li>
</ul>

<h2>Interactive behavior</h2>
<ul>
  <li><b>Mouse click on map:</b> selects a column.</li>
  <li><b>Mouse wheel on map:</b> changes the time step.</li>
  <li><b>Tab switch:</b> changes between heatmap and clustering view.</li>
</ul>

<h2>Interpretation notes</h2>
<ul>
  <li>Raw values preserve absolute magnitude.</li>
  <li>Normalized values are better for relative comparison.</li>
  <li>Clusters depend on the selected features and active time range.</li>
  <li>Columns with low or zero activity may behave differently depending on clustering settings.</li>
</ul>

<h2>Troubleshooting</h2>
<ul>
  <li>If nothing is displayed, check whether the folder structure is correct.</li>
  <li>If clustering looks odd, reduce or change the selected features.</li>
  <li>If a plot is empty, the selected region may have no values in the active time range.</li>
  <li>If labels are hard to read, enable column numbers and use contrasting colors.</li>
</ul>

</div>
"""

help_content = pn.pane.HTML(
    help_html,
    width=1000,
    height=700,
    sizing_mode="fixed",
)

def open_help(event=None):
    template.open_modal()

help_button.on_click(open_help)

template = pn.template.FastListTemplate(
    title="Column Explorer",
    header=[pn.Spacer(), help_button],
    sidebar=sidebar_content,
    sidebar_width=360,
    main=[main_content],
    accent_base_color="#2B7DE9",
    header_background="#000080",
    theme="default",
    theme_toggle=False,
)

template.modal.append(help_content)

refresh()
template.servable()