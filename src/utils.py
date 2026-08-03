import json
import sys
import h5py
import os
import pickle
import ants
import math
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import sobel, grey_dilation, center_of_mass


def parse_config():
    if len(sys.argv) > 1:
        filename = sys.argv[1]  
    else:
        filename = "default_config.json"
        
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        return config
    
    except FileNotFoundError:
        raise Exception(f"Error: Config file '{filename}' not found.")

    except json.JSONDecodeError as e:
        raise Exception(f"Error: Invalid JSON format in config file '{filename}': {e}")

def import_config():
    config = parse_config()

    # Make sure that paths terminate with /
    config["output_path"] = config["output_path"].rstrip("/") + "/"
    config["00_get_raw_images"]["input_path"] = config["00_get_raw_images"]["input_path"].rstrip("/") + "/"
        
    return config

def load_hdf5(path):
    def _parse(h5_obj):
        out = {}
        for key in h5_obj:
            item = h5_obj[key]
            if isinstance(item, h5py.Dataset):
                out[key] = {"data": item[()], "attrs": dict(item.attrs)}
            elif isinstance(item, h5py.Group):
                out[key] = {"children": _parse(item), "attrs": dict(item.attrs)}
        return out

    with h5py.File(path, 'r') as f:
        return _parse(f)

def crop_zyx_to_nonzero(arr):
    assert arr.ndim == 3, "Input must be a 3D array (Z, Y, X)"

    # Find non-zero indices along each axis
    z_nonzero = np.any(arr != 0, axis=(1, 2))
    y_nonzero = np.any(arr != 0, axis=(0, 2))
    x_nonzero = np.any(arr != 0, axis=(0, 1))

    # Find bounding box indices
    if not z_nonzero.any() or not y_nonzero.any() or not x_nonzero.any():
        # Entire array is zeros
        return arr[:0, :0, :0], ((0, 0), (0, 0), (0, 0))

    z_min, z_max = np.where(z_nonzero)[0][[0, -1]] + [0, 1]
    y_min, y_max = np.where(y_nonzero)[0][[0, -1]] + [0, 1]
    x_min, x_max = np.where(x_nonzero)[0][[0, -1]] + [0, 1]

    cropped_arr = arr[z_min:z_max, y_min:y_max, x_min:x_max]
    bbox = ((z_min, z_max), (y_min, y_max), (x_min, x_max))

    return cropped_arr, bbox

def resample_image(image, new_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkLinear):
    # Get the original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Compute new size based on new spacing
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    
    # Perform resampling
    resampled_image = resampler.Execute(image)
    
    return resampled_image

def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_safe(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_variable_json(data, path):
    with open(path, "w") as f:
        json.dump(_make_json_safe(data), f, indent=2)

def save_variable(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        
def load_variable(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def gradient_correlation(fixed_image, moving_image, mask=None):
    # Convert ANTsImage to numpy array
    fixed_np = fixed_image.numpy()
    moving_np = moving_image.numpy()

    # Compute gradients using Sobel operator
    grad_fixed_x = sobel(fixed_np, axis=0)
    grad_fixed_y = sobel(fixed_np, axis=1)
    grad_fixed_z = sobel(fixed_np, axis=2) if fixed_np.ndim == 3 else 0
    grad_moving_x = sobel(moving_np, axis=0)
    grad_moving_y = sobel(moving_np, axis=1)
    grad_moving_z = sobel(moving_np, axis=2) if moving_np.ndim == 3 else 0

    # Compute gradient magnitudes
    grad_fixed_mag = np.sqrt(grad_fixed_x**2 + grad_fixed_y**2 + grad_fixed_z**2)
    grad_moving_mag = np.sqrt(grad_moving_x**2 + grad_moving_y**2 + grad_moving_z**2)

    # Flatten arrays
    grad_fixed_flat = grad_fixed_mag.ravel()
    grad_moving_flat = grad_moving_mag.ravel()

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.numpy().ravel().astype(bool)
        grad_fixed_flat = grad_fixed_flat[mask_flat]
        grad_moving_flat = grad_moving_flat[mask_flat]

    # Remove any zeros or NaNs
    valid = np.isfinite(grad_fixed_flat) & np.isfinite(grad_moving_flat) & (grad_fixed_flat != 0) & (grad_moving_flat != 0)
    grad_fixed_flat = grad_fixed_flat[valid]
    grad_moving_flat = grad_moving_flat[valid]

    # Compute correlation
    gc = np.corrcoef(grad_fixed_flat, grad_moving_flat)[0, 1]
    return gc

# Translate image by translation vector
def translate_image(image, translation_vector):
    translation_vector_flipped = [-x for x in translation_vector]
    x_vox, y_vox, z_vox = tuple(translation_vector_flipped)
    x_spacing, y_spacing, z_spacing = image.spacing
    translation_vector_physical_units = (x_vox*x_spacing, y_vox*y_spacing, z_vox*z_spacing)

    # Create translation transform
    transform = ants.create_ants_transform(transform_type='Rigid3DTransform', translation=translation_vector_physical_units)
    
    # Apply transform
    translated_img = ants.apply_ants_transform_to_image(
        transform,
        image,
        image
    )

    return translated_img

def threshold_numpy(img, low, high):
    arr = img.numpy()
    mask = ((arr >= low) & (arr <= high)).astype(np.uint8)
    return ants.from_numpy(
        mask,
        origin=img.origin,
        spacing=img.spacing,
        direction=img.direction
    )
    
def get_sitk_image_from_point_cloud(point_cloud, org_img_size: tuple):
    array = np.zeros(org_img_size[::-1], dtype=np.uint8)

    for point in point_cloud:
        array[int(point[2]), int(point[1]), int(point[0])] = 1
    
    return sitk.GetImageFromArray(array)

def get_sitk_image_from_point_cloud_with_values(point_cloud, org_img_size: tuple):
    array = np.zeros(org_img_size[::-1], dtype=np.uint8)
    
    for point in point_cloud:
        z, y, x, value = point
        array[z, y, x] = value
        
    return sitk.GetImageFromArray(array)

def centers_to_single_voxels(inp_image):
    inp_arr = sitk.GetArrayFromImage(inp_image)
    
    centroid_img = sitk.Image(inp_image.GetSize(), sitk.sitkUInt8)
    centroid_img.CopyInformation(inp_image)
    
    for label in np.unique(inp_arr[inp_arr > 0]):
        coords = np.argwhere(inp_arr == label)  # (z, y, x)
        centroid = coords.mean(axis=0).astype(int)
        centroid_img[int(centroid[2]), int(centroid[1]), int(centroid[0])] = int(label)
    
    return centroid_img

def resample_to_isotropic(image, interpolator = sitk.sitkNearestNeighbor):
    spacing = np.array(image.GetSpacing())
    size = np.array(image.GetSize(), dtype=float)

    new_spacing = np.min(spacing)  # smallest voxel size → least loss
    new_spacing = [new_spacing] * 3

    new_size = np.round(size * (spacing / new_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def make_axis_image(reference_img, center, direction, length=500):
    arr = np.zeros(sitk.GetArrayFromImage(reference_img).shape, dtype=np.uint8)

    steps = np.linspace(-length/2, length/2, num=500)
    for s in steps:
        physical_point = np.array(center) + s * direction
        idx = reference_img.TransformPhysicalPointToIndex(physical_point)

        if all(0 <= idx[d] < arr.shape[::-1][d] for d in range(3)):
            arr[idx[2], idx[1], idx[0]] = 1

    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference_img)
    return img

def get_point_image(reference_img, physical_point):
    arr = np.zeros(sitk.GetArrayFromImage(reference_img).shape, dtype=np.uint8)

    idx = reference_img.TransformPhysicalPointToIndex(physical_point)
    arr[idx[2], idx[1], idx[0]] = 1

    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference_img)
    return img

def anisotropic_dilate(mask_ants, rx=0, ry=0, rz=0):
    arr = mask_ants.numpy()

    result = arr.copy()

    if rx > 0:
        sx = tuple([2*rx + 1, 1, 1])
        result = grey_dilation(result, size=sx)
    if ry > 0:
        sy = tuple([1, 2*ry + 1, 1])
        result = grey_dilation(result, size=sy)
    if rz > 0:
        sz = tuple([1, 1, 2*rz + 1])
        result = grey_dilation(result, size=sz)

    out = ants.from_numpy(
        result.astype(arr.dtype),
        origin=mask_ants.origin,
        spacing=mask_ants.spacing,
        direction=mask_ants.direction
    )
    
    return out

def get_label_centroids_image(img):
    df = ants.label_geometry_measures(img) 
    result = ants.image_clone(img) * 0
    
    for _, row in df.iterrows():
        pt = (row['Centroid_x'], row['Centroid_y'], row['Centroid_z'])
        
        idx = ants.transform_physical_point_to_index(img, pt)
        idx = tuple(int(round(i)) for i in idx)
        
        result[idx[0], idx[1], idx[2]] = int(row['Label'])
        
    return result

def get_label_centroids(mask_ants):
    df = ants.label_geometry_measures(mask_ants)
    
    centroids = {
        int(row['Label']): (row['Centroid_x'], row['Centroid_y'], row['Centroid_z'])
        for _, row in df.iterrows()
    }
    
    return centroids

def get_dilation_radii(spacing, dilation_factor):
    spacing = np.array(spacing, dtype=float)
    radii = dilation_factor * (min(spacing) / spacing)
    
    return tuple(int(round(r)) for r in radii)

def centroid_coords_to_image(coord_dict, reference_img):
    result = reference_img.clone() * 0
    
    for label, pt in coord_dict.items():
        idx = ants.transform_physical_point_to_index(reference_img, pt)
        idx = tuple(int(round(i)) for i in idx)
        result[idx[0], idx[1], idx[2]] = label
        
    return result

def get_point_image(reference_img, point):
    arr = np.zeros(sitk.GetArrayFromImage(reference_img).shape, dtype=np.uint8)

    arr[point[2], point[1], point[0]] = 1

    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(reference_img)
    return img

def print_sitk_metadata(image):
    print(f"Size:      {image.GetSize()}")
    print(f"Spacing:   {image.GetSpacing()}")
    print(f"Origin:    {image.GetOrigin()}")
    print(f"Direction: {image.GetDirection()}")
    print(f"Pixel type: {image.GetPixelIDTypeAsString()}")
    
def get_label_volumes(sitk_image):
    array = sitk.GetArrayFromImage(sitk_image)
    unique, counts = np.unique(array, return_counts=True)
    
    return dict(zip(unique.tolist(), counts.tolist()))

def round_up(x, rounding_number):
    return math.ceil(x / rounding_number) * rounding_number

def ransac_plane(points, n_iterations = 1000, distance_threshold = 10, seed = 42, print_output = False):
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(points), dtype=bool)
    best_count, best_normal, best_d = 0, 0, 0

    for n in range(n_iterations):
        # Pick 3 random points
        idx = rng.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal from cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        # Degenerate triangle: skip
        if norm < 1e-10:
            continue  
        
        # Normalize
        normal = normal / norm

        # Signed distance of every point to this plane
        d = -normal @ p1
        distances = np.abs(points @ normal + d)

        # Count inliers
        inliers = distances < distance_threshold
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if(print_output):
        print(f"Inliers: {best_inliers.sum()} / {len(points)} ({100 * best_inliers.mean():.1f}%)")
        print(f"Plane normal: {best_normal}")
        print(f"Plane offset d: {best_d:.4f}")

    return best_normal, best_d, best_inliers