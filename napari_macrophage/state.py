"""Shared state for the napari-macrophage plugin.

Exposes a single ``dataState`` instance of :class:`DataState` that all plugin
widgets read from and write to. Populated by :mod:`napari_macrophage.io` when
data is loaded, and consumed everywhere else (segmentation, analysis,
visualisation, mask editing).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from napari.utils.notifications import show_info


@dataclass
class DataState:
    """Container for the currently loaded image stacks, masks, and metadata.

    Attributes
    ----------
    cd206_images, dapi_images, collagen_images, F480_images : np.ndarray or None
        Per-channel 3D image stacks with shape ``(Z, Y, X)``.
    mask_images : np.ndarray or None
        Instance-label mask with shape ``(Z, Y, X)`` and dtype ``uint8``.
    mask_path : Path or None
        Path the mask was last loaded from (used for save-back workflows).
    file_name : str or None
        Stem of the currently loaded image file, used to name exports.
    voxel_size_um : tuple of float or None
        Physical voxel size ``(z, y, x)`` in micrometres. Required for
        morphology metrics and isotropic resampling.
    """

    cd206_images: np.ndarray | None = None
    dapi_images: np.ndarray | None = None
    mask_images: np.ndarray | None = None
    collagen_images: np.ndarray | None = None
    F480_images: np.ndarray | None = None
    mask_path: Path | None = None
    file_name: str | None = None
    voxel_size_um: tuple[float, float, float] | None = None


dataState = DataState()


def set_voxel_size_um(voxel_x: float, voxel_y: float, voxel_z: float):
    """Set the voxel size on the global :data:`dataState`.

    Parameters
    ----------
    voxel_x, voxel_y, voxel_z : float
        Physical size of a voxel along each axis, in micrometres. The values
        are stored internally as ``(z, y, x)`` to match array axis order.
    """
    dataState.voxel_size_um = (voxel_z, voxel_y, voxel_x)
    show_info(f"Voxel size set to: x={voxel_x} µm, y={voxel_y} µm, z={voxel_z} µm")


def get_voxel_size_um() -> tuple[float, float, float] | None:
    """Return the current voxel size ``(z, y, x)`` in µm, or ``None`` if unset."""
    return dataState.voxel_size_um
