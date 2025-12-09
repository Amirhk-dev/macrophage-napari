from dataclasses import dataclass
from pathlib import Path
import numpy as np
from napari.utils.notifications import show_info

@dataclass
class DataState:
    cd206_images: np.ndarray | None = None # store the original images and masks
    dapi_images: np.ndarray | None = None
    mask_images: np.ndarray | None = None
    collagen_images: np.ndarray | None = None
    F480_images: np.ndarray | None = None
    mask_path: Path | None = None
    # mask_layer: Napari_Labels | None = None # store mask layer object
    file_name: str | None = None
    voxel_size_um: tuple[float, float, float] | None = None  # (x, y, z) in µm

dataState = DataState()

def set_voxel_size_um(voxel_x: float, voxel_y: float, voxel_z: float):
    dataState.voxel_size_um = (voxel_z, voxel_y, voxel_x)
    show_info(f"Voxel size set to: x={voxel_x} µm, y={voxel_y} µm, z={voxel_z} µm")

def get_voxel_size_um() -> tuple[float, float, float]|None:
    return dataState.voxel_size_um