import napari
import numpy as np
import tifffile as tiff
import re
import zarr

from pathlib import Path
from napari.utils.notifications import show_info, show_warning
from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import QFileDialog, QMessageBox
from napari.layers import Image as Napari_Image, Labels as Napari_Labels

from .state import dataState
from .edit_mask_image import select_object

###### upload image and mask ######
def _add_or_update_channel(viewer, image: np.ndarray, name: str, colormap: str, visible: bool = False):
    if name in viewer.layers:
        viewer.layers[name].data = image
    else:
        layer = viewer.add_image(
            image,
            name=name,
            blending="additive",
            colormap=colormap
        )
        layer.visible = visible
        viewer.dims.current_step = (0, 0)


def add_image_layer(image_path: Path = Path("/Users/tianru.li/Desktop/data/image/image_1_with_dapi.tif")):
    img = tiff.imread(image_path)
    with_dapi = True
    if img.ndim == 3:
        dataState.cd206_images = img
        with_dapi = False
    elif img.ndim == 4 and min(img.shape) == 2:
        min_dim = np.argmin(img.shape)
        img_reordered = np.moveaxis(img, min_dim, 0)
        dataState.cd206_images = img_reordered[0]
        dataState.dapi_images = img_reordered[1]
    elif img.ndim == 4 and min(img.shape) == 5:
        print(img.shape)
        min_dim = np.argmin(img.shape)
        img_reordered = np.moveaxis(img, min_dim, 0)
        dataState.cd206_images = img_reordered[2]
        dataState.dapi_images = img_reordered[3]
        dataState.collagen_images = img_reordered[0]
        dataState.F480_images = img_reordered[1]
        print(img_reordered.shape)
    # else:
    #     print(img.shape, img.ndim)

    viewer = napari.current_viewer()
    if dataState.cd206_images is not None:
        _add_or_update_channel(viewer, dataState.cd206_images, name="CD206", colormap="magenta", visible=True)
        dataState.file_name = image_path.stem
        # parts = dataState.file_name.split("_")
        # dataState.file_name = "_".join(parts[:2])
        print("file name", dataState.file_name)

    if dataState.dapi_images is not None:
        _add_or_update_channel(viewer, dataState.dapi_images, name="DAPI", colormap="cyan", visible=True)

    if dataState.collagen_images is not None:
        _add_or_update_channel(viewer, dataState.collagen_images, name="Collagen", colormap="blue", visible=False)

    if dataState.F480_images is not None:
        _add_or_update_channel(viewer, dataState.F480_images, name="F480", colormap="green", visible=False)

    if with_dapi:
        msg = f"Loaded macrophage image: {image_path}"
        show_info(msg)
    else:
        msg = f"Loaded image without DAPI channel: {image_path}"
        show_info(msg)


def add_mask_layer(mask_path: Path = Path("/Users/tianru.li/Desktop/data/mask/image_1.tif")):
    dataState.mask_path = mask_path
    mask = tiff.imread(mask_path)
    if mask.ndim != 3:
        msg = "Please upload mask of shape (z, y, x)"
        show_warning(msg) 
    else:
        dataState.mask_images = mask.astype(np.uint8) 
    # dataState.file_name = mask_path.stem

    viewer = napari.current_viewer()
    if "Masks" in viewer.layers:
        viewer.layers["Masks"].data = dataState.mask_images
    else:
        viewer.add_labels(dataState.mask_images, name="Masks")
        viewer.dims.current_step = (0,)
    if select_object not in getattr(viewer.layers["Masks"], "mouse_drag_callbacks", []):
        viewer.layers["Masks"].mouse_drag_callbacks.append(select_object)
    if not hasattr(viewer.layers["Masks"], "selected_object_id"):
        viewer.layers["Masks"].selected_object_id = None
    if not hasattr(viewer.layers["Masks"], "click_coords"):
        viewer.layers["Masks"].click_coords = None
        
    msg = f"Loaded mask: {mask_path}"
    show_info(msg)
    print(msg)


def _load_image_mask_from_zarr_group(viewer, root):
    img_arr = None
    if "image" in root:
        try:
            img_arr = np.asarray(root["image"])
        except Exception as e:
            show_warning(f"Failed to read image from zarr: {e}")

    mask_arr = None
    if "mask" in root:
        try:
            mask_arr = np.asarray(root["mask"])
        except Exception as e:
            show_warning(f"Failed to read mask from zarr: {e}")

    if img_arr is not None:
        if img_arr.ndim == 4 and img_arr.shape[0] == 2:
            dataState.cd206_images = img_arr[0]
            dataState.dapi_images = img_arr[1]
        elif img_arr.ndim == 3:
            dataState.cd206_images = img_arr
        else:
            show_warning(f"Unsupported image shape: {img_arr.shape}")

    if mask_arr is not None:
        if mask_arr.ndim == 3:
            dataState.mask_images = mask_arr.astype(np.uint8)
        else:
            show_warning(f"Unsupported mask shape: {mask_arr.shape}")

    if "bboxes2d" in root:
        reply = QMessageBox.question(
            None,
            "Import ROIs from zarr?",
            "Detected 'bboxes2d' in zarr. Import as ROI rectangles?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            bb = root["bboxes2d"]
            try:
                data = np.asarray(bb["data"])  # rows: [z, label_id, ymin, xmin, ymax, xmax]
            except Exception as e:
                data = None
                show_warning(f"Failed to read bboxes2d: {e}")
            if data is not None and data.size > 0:
                if "ROI" not in viewer.layers:
                    roi_layer = viewer.add_shapes(
                        name="ROI",
                        shape_type="rectangle",
                        edge_color="red",
                        face_color="transparent",
                        edge_width=2
                    )
                else:
                    roi_layer = viewer.layers["ROI"]

                blocker = getattr(roi_layer.events.data, "blocker", None)
                if blocker is not None:
                    with roi_layer.events.data.blocker():
                        for row in data:
                            z, label_id, ymin, xmin, ymax, xmax = row.tolist()
                            curr_bbox = np.array([
                                [z, ymin, xmin],
                                [z, ymin, xmax],
                                [z, ymax, xmax],
                                [z, ymax, xmin],
                            ], dtype=float)
                            roi_layer.add(
                                curr_bbox,
                                shape_type="rectangle",
                                edge_color="red",
                                face_color="transparent",
                                edge_width=2
                            )
            # print("data",data)
            # print("roi data", roi_layer.data)
            show_info(f"Imported {data.shape[0]} ROIs from zarr")

    return dataState.cd206_images, dataState.dapi_images, dataState.mask_images


def add_layer_from_zarr(folder: Path | None = None):
    if folder is None:
        dir_path = QFileDialog.getExistingDirectory(None, "Select .zarr folder", "")
        if not dir_path:
            return
        folder = Path(dir_path)
    else:
        folder = Path(folder)

    try:
        root = zarr.open_group(str(folder), mode="r")
        viewer = napari.current_viewer()
        cd206_images, dapi_images, mask_images = _load_image_mask_from_zarr_group(viewer, root)
    except Exception as e:
        show_warning(f"Failed to open zarr: {e}")
        return 
    file_name = folder.stem
    # parts = file_name.split("_")
    # file_name = "_".join(parts[:2])  # image_1
    dataState.file_name = file_name
    # print("file name", file_name)

    if cd206_images is not None:
        if "CD206" in viewer.layers:
            viewer.layers["CD206"].data = cd206_images
        else:
            viewer.add_image(cd206_images, name="CD206", blending="additive", colormap="magenta")

    if dapi_images is not None:
        if "DAPI" in viewer.layers:
            viewer.layers["DAPI"].data = dapi_images
        else:
            viewer.add_image(dapi_images, name="DAPI", blending="additive", colormap="cyan")

    if mask_images is not None:
        if "Masks" in viewer.layers:
            viewer.layers["Masks"].data = mask_images
            viewer.layers["Masks"]
        else:
            viewer.add_labels(mask_images, name="Masks")


def _is_label_layer(arr):
    a = np.asarray(arr)
    if a.ndim != 3:
        return False
    if np.issubdtype(a.dtype, np.integer): # if int64/ uint8...
        return True
    if np.issubdtype(a.dtype, np.floating):
        frac = np.modf(a)[0] # separate a to fractional [0] and integer [1] parts
        return np.all(frac == 0)
    return False


def _prepare_opened_image_and_mask(viewer, layer):
    """ Process files uploaded via the default Open File (not the customised load image/ mask).
    This function extracts the correct channels from the uploaded image and create corresponding layers in the layer list."""
    src = getattr(layer, "source", None)
    file_path = getattr(src, "path", None)
   
    # first check if it is a zarr file
    zarr_root = None
    if isinstance(file_path, str):
        m = re.search(r"(.*\.zarr)(?:/.*)?$", file_path)
        if m:
            zarr_root = m.group(1)
            # print("zarr yes", zarr_root)

    if zarr_root:
        try:
            root = zarr.open_group(zarr_root, mode="r")
            cd206_images, dapi_images, mask_images = _load_image_mask_from_zarr_group(viewer, root)
        except Exception as e:
            show_warning(f"Failed to open zarr: {e}")
            return

        file_name = Path(zarr_root).stem  # e.g. image_1_with_dapi
        # parts = file_name.split("_")
        # file_name = "_".join(parts[:2])  # image_1
        dataState.file_name = file_name

        if cd206_images is not None:
            if "CD206" in viewer.layers:
                viewer.layers["CD206"].data = cd206_images
            else:
                viewer.add_image(cd206_images, name="CD206", blending="additive", colormap="magenta")
        if dapi_images is not None:
            if "DAPI" in viewer.layers:
                viewer.layers["DAPI"].data = dapi_images
            else:
                viewer.add_image(dapi_images, name="DAPI", blending="additive", colormap="cyan")
        if mask_images is not None:
            if "Masks" in viewer.layers:
                viewer.layers["Masks"].data = mask_images
            else:
                viewer.add_labels(mask_images, name="Masks")
            if select_object not in getattr(viewer.layers["Masks"], "mouse_drag_callbacks", []):
                viewer.layers["Masks"].mouse_drag_callbacks.append(select_object)
            if not hasattr(viewer.layers["Masks"], "selected_object_id"):
                viewer.layers["Masks"].selected_object_id = None
            if not hasattr(viewer.layers["Masks"], "click_coords"):
                viewer.layers["Masks"].click_coords = None
        try:
            viewer.layers.remove(layer)
        except Exception:
            pass
        return
    
    # if not, process as tiff
    if file_path:
        file_name = Path(file_path).stem
        # parts = file_name.split("_")
        # file_name = "_".join(parts[:2]) # image_1
        dataState.file_name = file_name
        # print("file name", file_name)

    layer_name = getattr(layer, "name", "")
    if layer_name.startswith("Object ") or layer_name in ("Preview Mask", "ROI", "Masks", "CD206", "DAPI"):
        return

    if isinstance(layer, (Napari_Image, Napari_Labels)) and _is_label_layer(layer.data):
        dataState.mask_images = layer.data.astype(np.uint8).copy()
        if "Masks" not in viewer.layers:
            viewer.add_labels(dataState.mask_images, name="Masks")
        else:
            viewer.layers["Masks"].data = dataState.mask_images
        if select_object not in getattr(viewer.layers["Masks"], "mouse_drag_callbacks", []):
            viewer.layers["Masks"].mouse_drag_callbacks.append(select_object)
        if not hasattr(viewer.layers["Masks"], "selected_object_id"):
            viewer.layers["Masks"].selected_object_id = None
        if not hasattr(viewer.layers["Masks"], "click_coords"):
            viewer.layers["Masks"].click_coords = None
        try:
            viewer.layers.remove(layer)
        except Exception:
            pass
    
    elif isinstance(layer, Napari_Image):
        if layer.data.ndim == 4 and layer.data.shape[0] == 2:
            dataState.cd206_images = layer.data[0]
            dataState.dapi_images = layer.data[1]
            if "CD206" not in viewer.layers:
                viewer.add_image(dataState.cd206_images, name="CD206", blending="additive")
            else:
                viewer.layers["CD206"].data = dataState.cd206_images
            if "DAPI" not in viewer.layers:
                viewer.add_image(dataState.dapi_images, name="DAPI", blending="additive")
            else:
                viewer.layers["DAPI"].data = dataState.dapi_images
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass
        elif layer.data.ndim == 3:
            if "CD206" not in viewer.layers:
                dataState.cd206_images = layer.data
                viewer.add_image(dataState.cd206_images, name="CD206", blending="additive")
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass


def _prepare_all_layers(viewer):
    for l in list(viewer.layers):
        _prepare_opened_image_and_mask(viewer, l)

