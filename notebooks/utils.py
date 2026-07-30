"""
This file contains all additional functions used by the main notebooks
"""

import zipfile
import tempfile
import tifffile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roifile import ImagejRoi
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.color import gray2rgb


def strip_roi_suffix(stem):
    """
    To match all potential suffix used to save the ROI zip files
    """
    for suffix in ("_roiset", "_rois", "_roi"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def match_images_to_rois(raw_image_dir, roi_dir):
    """
    Pair raw images with their ROI zip based on filename alone: an image
    "<name>.tif" is matched to a ROI zip named "<name>_RoiSet.zip" (or
    "<name>.zip"). If the ROI file carries an extra prefix, the image stem
    still matches whenever the ROI base name is contained in it.
    """
    image_paths = sorted(
        p for p in raw_image_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")
    )
    roi_paths = sorted(p for p in roi_dir.iterdir() if p.suffix.lower() == ".zip")

    roi_by_base_name = {strip_roi_suffix(p.stem): p for p in roi_paths}

    samples = []
    unmatched = []

    for image_path in image_paths:
        roi_path = roi_by_base_name.get(image_path.stem)

        if roi_path is None:
            candidates = [
                p for base_name, p in roi_by_base_name.items()
                if base_name.lower() in image_path.stem.lower()
            ]
            if len(candidates) == 1:
                roi_path = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous ROI match for {image_path.name}: "
                    f"{[c.name for c in candidates]}"
                )

        if roi_path is None:
            unmatched.append(image_path)
        else:
            samples.append({
                "sample_id": image_path.stem,
                "image_path": image_path,
                "roi_zip_path": roi_path,
            })

    if unmatched:
        print("Warning: no matching ROI zip found for:", [p.name for p in unmatched])

    return pd.DataFrame(samples)


def normalize01_within_mask(image, mask):
    image = image.astype(float)
    vals = image[mask]

    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)

    image_norm = (image - lo) / (hi - lo + 1e-10)
    image_norm = np.clip(image_norm, 0, 1)

    return image_norm


def normalize01(img):
    """
    Normalize image for display only.
    """
    img = img.astype(float)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1 + 1e-10)


def get_roi_by_name(rois, keyword):
    """
    Find the correct ROI in a zip file
    """
    matches = [
        roi for roi in rois
        if keyword.lower() in roi.name.lower()
    ]

    if len(matches) == 0:
        raise ValueError(f"No ROI found containing: {keyword}")

    return matches[0]


def roi_to_mask(roi, shape):
    """
    Creates a binary mask based on the .roi
    Outputs 1 if in the ROI, 0 otherwise
    """
    coords = roi.coordinates()
    x, y = coords[:, 0], coords[:, 1]

    rr, cc = polygon(y, x, shape=shape)

    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def densify_surface_coords(coords, spacing=2):
    """
    Interpolate extra points along each segment of a fragmented line 
    so consecutive points are ~spacing px apart.
    """
    coords = coords.astype(float)
    dense_points = []

    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]

        dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        n_points = max(int(dist / spacing), 2)

        xs = np.linspace(x0, x1, n_points)
        ys = np.linspace(y0, y1, n_points)

        dense_points.append(np.column_stack([xs, ys]))

    return np.vstack(dense_points)


def surface_tangent_angles(surface_roi, spacing=2):
    """
    Returns the tangent angle to the inner cortical surface
    """
    raw_coords = surface_roi.coordinates().astype(float)

    coords = densify_surface_coords(raw_coords, spacing=spacing)

    x = coords[:, 0]
    y = coords[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)

    tangent_angles = np.degrees(np.arctan2(dy, dx))

    return coords, tangent_angles


def nearest_surface_reference(pixel_y, pixel_x, surface_coords, tangent_angles):
    """
    Find the nearest reference on the inner surface

    Outputs the closest x and y coordinates, the local tangent and radial angles
    """
    surface_x = surface_coords[:, 0]
    surface_y = surface_coords[:, 1]

    px = pixel_x[:, None]
    py = pixel_y[:, None]

    distances = (px - surface_x[None, :])**2 + (py - surface_y[None, :])**2
    nearest_idx = np.argmin(distances, axis=1)

    nearest_x = surface_x[nearest_idx]
    nearest_y = surface_y[nearest_idx]

    local_tangent = tangent_angles[nearest_idx]
    local_radial = local_tangent + 90

    return nearest_x, nearest_y, local_tangent, local_radial


def angle_difference_deg(a, b):
    diff = np.abs(a - b)
    diff = np.mod(diff, 180)
    diff = np.minimum(diff, 180 - diff)
    return diff


def prepare_image_for_cellpose(img):
    """
    Prepare image for Cellpose-SAM while keeping the original image for display.
    Cellpose can handle 2D grayscale images directly.
    """
    if img.ndim in (2, 3):
        return img

    raise ValueError(f"Unsupported image shape: {img.shape}")


def image_for_display(img):
    """
    Return a 2D image for grayscale display.
    """
    if img.ndim == 2:
        return normalize01(img)
    if img.ndim == 3:
        return normalize01(img.mean(axis=2))
    raise ValueError(f"Unsupported image shape for display: {img.shape}")


def roi_to_mask_from_roi(roi, image_shape):
    """
    Convert an ImageJ polygon ROI into a boolean mask.
    image_shape should be 2D: (height, width).
    """
    coords = roi.coordinates()
    x = coords[:, 0]
    y = coords[:, 1]

    mask = np.zeros(image_shape, dtype=bool)
    rr, cc = polygon(y, x, shape=image_shape)
    mask[rr, cc] = True

    return mask


def load_rois(roi_path):
    """
    Load ROIs from a .zip RoiSet, a folder of .roi files, or a single .roi
    file. Returns a dict {roi_name: ImagejRoi}.
    """
    roi_path = Path(roi_path)

    rois = {}

    if roi_path.is_dir():
        roi_files = sorted(roi_path.glob("*.roi"))
        if len(roi_files) == 0:
            raise FileNotFoundError(f"No .roi files found in {roi_path}")
        for f in roi_files:
            roi = ImagejRoi.fromfile(f)
            name = roi.name if roi.name else f.stem
            rois[name] = roi

    elif roi_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(roi_path, "r") as zf:
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                roi_files = sorted(Path(tmpdir).glob("*.roi"))
                if len(roi_files) == 0:
                    raise FileNotFoundError(f"No .roi files found in {roi_path}")
                for f in roi_files:
                    roi = ImagejRoi.fromfile(f)
                    name = roi.name if roi.name else f.stem
                    rois[name] = roi

    elif roi_path.suffix.lower() == ".roi":
        roi = ImagejRoi.fromfile(roi_path)
        name = roi.name if roi.name else roi_path.stem
        rois[name] = roi

    else:
        raise ValueError(f"Unsupported ROI path: {roi_path}")

    return rois


def find_roi_by_keywords(rois, keywords):
    """
    Find first ROI whose name contains one of the requested keywords.
    """
    available = list(rois.keys())

    for keyword in keywords:
        for name, roi in rois.items():
            if keyword.lower() in name.lower():
                return name, roi

    raise ValueError(
        f"Could not find ROI with keywords {keywords}. "
        f"Available ROI names: {available}"
    )


def count_objects_in_roi(masks, roi_mask):
    """
    Count Cellpose objects whose centroid falls inside roi_mask.
    One detected Cellpose mask = one object; counted if its centroid is
    inside the ROI.
    """
    count = 0
    object_ids = []
    object_rows = []

    for region in regionprops(masks):
        y, x = region.centroid

        if roi_mask[int(y), int(x)]:
            count += 1
            object_ids.append(region.label)
            object_rows.append({
                "object_id": int(region.label),
                "centroid_x": float(x),
                "centroid_y": float(y),
                "area_px": int(region.area),
                "bbox_min_row": int(region.bbox[0]),
                "bbox_min_col": int(region.bbox[1]),
                "bbox_max_row": int(region.bbox[2]),
                "bbox_max_col": int(region.bbox[3]),
            })

    return count, object_ids, object_rows


def make_count_overlay(display_img, masks, right_mask, left_mask, right_ids, left_ids, title, out_path):
    """
    Save overlay showing the image in grayscale, ROI contours, counted
    right-side objects in red, and counted left-side objects in cyan.
    """
    base = image_for_display(display_img)
    rgb = gray2rgb(base)

    right_counted = np.isin(masks, right_ids)
    left_counted = np.isin(masks, left_ids)

    right_boundaries = find_boundaries(right_counted, mode="outer")
    left_boundaries = find_boundaries(left_counted, mode="outer")

    plt.figure(figsize=(12, 10))
    plt.imshow(rgb)

    plt.imshow(np.ma.masked_where(~right_counted, right_counted), alpha=0.25, cmap="Reds")
    plt.imshow(np.ma.masked_where(~left_counted, left_counted), alpha=0.25, cmap="winter")

    plt.contour(right_mask, levels=[0.5], colors="red", linewidths=2)
    plt.contour(left_mask, levels=[0.5], colors="cyan", linewidths=2)

    plt.contour(right_boundaries, levels=[0.5], colors="red", linewidths=0.5)
    plt.contour(left_boundaries, levels=[0.5], colors="cyan", linewidths=0.5)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def make_numbered_overlay(display_img, masks, roi_mask, object_ids, title, out_path, color="red", MAX_LABELS_TO_DRAW=500):
    """
    Overlay with sequential numbers for objects inside one ROI.
    Allows to see the first segmented fibers and counting
    """
    base = image_for_display(display_img)

    object_id_set = set(object_ids)
    roi_regions = [r for r in regionprops(masks) if r.label in object_id_set]

    plt.figure(figsize=(12, 10))
    plt.imshow(base, cmap="gray")
    plt.contour(roi_mask, levels=[0.5], colors=color, linewidths=2)

    for i, region in enumerate(roi_regions[:MAX_LABELS_TO_DRAW], start=1):
        y, x = region.centroid
        plt.text(x, y, str(i), color=color, fontsize=5, ha="center", va="center")

    suffix = ""
    if len(roi_regions) > MAX_LABELS_TO_DRAW:
        suffix = f" (showing first {MAX_LABELS_TO_DRAW})"

    plt.title(f"{title} — counted objects: {len(roi_regions)}{suffix}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def resolve_path(path_value, base_dir=None):
    """Resolve an absolute path, or a path relative to PROJECT_DIR / base_dir."""
    p = Path(str(path_value))
    candidates = [p] if p.is_absolute() else [PROJECT_DIR / p]
    if base_dir is not None and not p.is_absolute():
        candidates.append(base_dir / p)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve path: {path_value}")


def get_line_endpoints(roi):
    """
    Return (x1, y1, x2, y2) for a line ROI.

    Find the two farthest point in the segmented line to separate a ROI to know how it is split.
    """
    coords = roi.coordinates().astype(float)
    if coords.shape[0] < 2:
        raise ValueError(f"Expected at least 2 points for line ROI '{roi.name}', got {coords.shape[0]}")
    if coords.shape[0] == 2:
        (x1, y1), (x2, y2) = coords
        return x1, y1, x2, y2
    diffs = coords[:, None, :] - coords[None, :, :]
    dist2 = (diffs ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    x1, y1 = coords[i]
    x2, y2 = coords[j]
    return x1, y1, x2, y2


def line_y_at_x(x, x1, y1, x2, y2):
    """
    y-coordinate of the (x1,y1)-(x2,y2) line at position x.

    Fully vectorized/elementwise: x, x1, y1, x2, y2 can each be a scalar or an
    array (e.g. one row per fiber object, each with its own line).
    """
    x = np.asarray(x, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    dx = x2 - x1
    safe_dx = np.where(dx != 0, dx, 1.0)
    slope = (y2 - y1) / safe_dx

    return np.where(dx != 0, y1 + slope * (x - x1), (y1 + y2) / 2)


def split_roi_area(cpu_roi, sep_roi, image_shape):
    """
    Turns the striatum ROI into a mask and separate it based on the drawn separation ROI
    Classify between dorsal and ventral region
    """
    roi_mask = roi_to_mask(cpu_roi, image_shape)
    x1, y1, x2, y2 = get_line_endpoints(sep_roi)

    height, width = image_shape
    cols = np.arange(width)
    line_y_per_col = line_y_at_x(cols, x1, y1, x2, y2)
    rows = np.arange(height)[:, None]

    dorsal_mask = roi_mask & (rows < line_y_per_col[None, :])
    dorsal_area = int(dorsal_mask.sum())
    total_area = int(roi_mask.sum())
    ventral_area = total_area - dorsal_area

    return dorsal_area, ventral_area, total_area


def plot_dorsal_ventral_qc(samples, sample_id, objects_df, downsample=4, figsize=(9, 8)):
    """
    QC overlay: raw image + Striatum_R/Striatum_L outline + sep_R/sep_L line + classified fiber centroids.
    """
    match = samples[samples["sample_id"].astype(str) == str(sample_id)]
    if match.empty:
        raise ValueError(f"No sample found for sample_id={sample_id!r}")
    sample = match.iloc[0]

    image_path = sample["image_path"]
    roi_path = sample["roi_zip_path"]
    image_file = image_path.name

    rois = ImagejRoi.fromfile(roi_path)
    cpu_r = get_roi_by_name(rois, "Striatum_R")
    cpu_l = get_roi_by_name(rois, "Striatum_L")
    sep_r = get_roi_by_name(rois, "sep_R")
    sep_l = get_roi_by_name(rois, "sep_L")

    img = tifffile.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    height, width = img.shape
    display_img = normalize01(img[::downsample, ::downsample])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display_img, cmap="gray", extent=[0, width, height, 0])

    for roi in (cpu_r, cpu_l):
        coords = roi.coordinates()
        coords = np.vstack([coords, coords[:1]])  # close the polygon
        ax.plot(coords[:, 0], coords[:, 1], color="yellow", linewidth=1.5)

    for sep_roi in (sep_r, sep_l):
        x1, y1, x2, y2 = get_line_endpoints(sep_roi)
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=2, linestyle="--")

    fibers = objects_df[
        (objects_df["sample_id"].astype(str) == str(sample_id))
        & (objects_df["image_file"] == image_file)
    ]
    region_colors = {"dorsal": "red", "ventral": "deepskyblue"}
    for region, color in region_colors.items():
        pts = fibers[fibers["region"] == region]
        ax.scatter(pts["centroid_x"], pts["centroid_y"], s=4, color=color, label=region, alpha=0.8)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f"{sample_id}\nStriatum_R/Striatum_L outline (yellow), sep line (white)")
    ax.legend(loc="upper right", markerscale=3)
    ax.axis("off")
    fig.tight_layout()
    return fig

