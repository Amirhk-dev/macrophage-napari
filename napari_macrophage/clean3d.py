"""Per-object 3D mask cleanup for the Masks layer.

Cleans each labeled macrophage independently: per-slice erosion +
largest-2D-component + fill-holes + closing + Gaussian smoothing, followed by
an optional 3D closing and largest-3D-connected-component with a minimum
voxel threshold.
"""

from __future__ import annotations

import napari
import numpy as np
from napari.utils.notifications import show_info, show_warning
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology

from .error import _layers_not_in_viewer_error


def clean_mask_per_slice(
    masks_by_z: dict,
    erosion_radius: int = 3,
    closing_radius: int = 3,
    gaussian_sigma: float = 1.5,
) -> dict:
    """Erode + keep-largest + fill holes + close + smooth, per z-slice."""
    cleaned: dict = {}

    for z, mask in masks_by_z.items():
        m = mask.astype(bool)

        eroded = morphology.binary_erosion(m, morphology.disk(erosion_radius))
        labeled = measure.label(eroded)
        if labeled.max() == 0:
            m_clean = m
        else:
            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            m_clean = (labeled == largest.label)

        m_clean = ndimage.binary_fill_holes(m_clean)

        if closing_radius > 0:
            m_clean = morphology.binary_closing(m_clean, morphology.disk(closing_radius))

        if gaussian_sigma > 0:
            smoothed = gaussian_filter(m_clean.astype(float), sigma=gaussian_sigma)
            m_clean = smoothed >= 0.5

        labeled = measure.label(m_clean)
        if labeled.max() > 0:
            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            m_clean = (labeled == largest.label)

        cleaned[z] = m_clean.astype(bool)

    return cleaned


def largest_3d_component(
    masks_by_z: dict,
    closing_radius_3d: int = 0,
    min_voxels: int = 0,
) -> dict:
    """Keep only the largest 26-connected component across the stacked slices.

    Optionally applies a 3D binary closing (ball structuring element) before
    connected-component analysis, and drops the object entirely if the largest
    component is smaller than ``min_voxels``.
    """
    if not masks_by_z:
        return masks_by_z

    z_indices = sorted(masks_by_z.keys())
    z_min = z_indices[0]

    sample = masks_by_z[z_indices[0]]
    H, W = sample.shape

    n_z = z_indices[-1] - z_min + 1
    vol = np.zeros((n_z, H, W), dtype=bool)
    for z in z_indices:
        vol[z - z_min] = masks_by_z[z].astype(bool)

    if closing_radius_3d > 0:
        vol = ndimage.binary_closing(vol, structure=morphology.ball(closing_radius_3d))

    labeled, num_features = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    if num_features == 0:
        return {}

    component_sizes = ndimage.sum(vol, labeled, range(1, num_features + 1))
    largest_idx = int(np.argmax(component_sizes))
    largest_size = int(component_sizes[largest_idx])
    if min_voxels > 0 and largest_size < min_voxels:
        return {}

    largest_vol = (labeled == (largest_idx + 1))

    cleaned = {}
    for z in range(n_z):
        slice_mask = largest_vol[z]
        if slice_mask.any():
            cleaned[z + z_min] = slice_mask

    return cleaned


def clean_3d_macrophages(
    erosion_radius: int = 3,
    closing_radius_2d: int = 3,
    gaussian_sigma: float = 1.5,
    closing_radius_3d: int = 0,
    min_voxels_3d: int = 500,
):
    """Clean every labeled macrophage in the Masks layer independently.

    For each object ID: extract per-z slices, run per-slice cleanup, then
    optional 3D closing and largest-connected-component with a min-voxels
    drop threshold. Original IDs are preserved. When cleaned masks of
    different objects overlap (possible after closing), the higher object
    ID wins for that voxel.
    """
    viewer = napari.current_viewer()
    if _layers_not_in_viewer_error(viewer, ["Masks"]):
        return

    masks = viewer.layers["Masks"].data
    if masks.ndim != 3:
        show_warning(f"Masks must be 3D (Z, Y, X); got shape {masks.shape}.")
        return

    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids > 0]
    if unique_ids.size == 0:
        show_info("No objects to clean.")
        return

    show_info(
        f"Cleaning {unique_ids.size} object(s) — 2D erosion={erosion_radius}, "
        f"2D closing={closing_radius_2d}, sigma={gaussian_sigma}, "
        f"3D closing={closing_radius_3d}, min voxels={min_voxels_3d}"
    )

    Z = masks.shape[0]
    cleaned_vol = np.zeros_like(masks)

    dropped = 0
    for obj_id in unique_ids:
        obj_mask = (masks == obj_id)
        z_present = np.where(obj_mask.any(axis=(1, 2)))[0]
        if z_present.size == 0:
            continue

        masks_by_z = {int(z): obj_mask[z] for z in z_present}

        cleaned = clean_mask_per_slice(
            masks_by_z,
            erosion_radius=int(erosion_radius),
            closing_radius=int(closing_radius_2d),
            gaussian_sigma=float(gaussian_sigma),
        )
        cleaned = largest_3d_component(
            cleaned,
            closing_radius_3d=int(closing_radius_3d),
            min_voxels=int(min_voxels_3d),
        )

        if not cleaned:
            dropped += 1
            continue

        for z, m in cleaned.items():
            if 0 <= z < Z:
                cleaned_vol[z][m] = obj_id

    viewer.layers["Masks"].data = cleaned_vol
    viewer.layers["Masks"].refresh()

    if dropped:
        show_warning(f"Cleaned {unique_ids.size - dropped} object(s); {dropped} removed (empty or below min voxels).")
    else:
        show_info(f"Cleaned {unique_ids.size} object(s).")
