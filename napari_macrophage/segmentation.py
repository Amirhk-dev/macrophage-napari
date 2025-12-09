import os
import napari
import numpy as np

from pathlib import Path
from magicgui import magicgui
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QMessageBox


from scipy.ndimage import label, binary_opening, binary_closing, distance_transform_edt
from skimage.measure import label as sk_label, regionprops
from skimage.feature import peak_local_max
from skimage.draw import polygon2mask
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from dataclasses import dataclass

from .state import dataState
from .error import _layers_not_in_viewer_error, _layer_not_in_viewer_error
from .edit_mask_image import select_object

@dataclass
class OtsuState:
    image_3d: np.ndarray | None = None          # volume inside bbox (first None: allows NoneType; second None: initialise to None)
    bbox_mask_3d: np.ndarray | None = None      # 3D bbox mask by copying the 2D bbox to all slices from z
    last_bbox_z: int | None = None              # slice index where bbox drawn
    last_bbox_xy: np.ndarray | None = None      # (4, 2) bbox in (y,x)
    threshold: float | None = None              # current otsu threshold
    preview_mode: str | None = None             # "otsu" or "watershed"

###### run automatic segmentation by adding a bounding box ######
def _get_otsu_state(viewer) -> OtsuState:
    required_layers = ["CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        raise RuntimeError("Image layer for running segmentation not found")
    img_layer = viewer.layers["CD206"]
    key = "_macrophage_otsu_state"
    if not hasattr(img_layer, key):
        setattr(img_layer, key, OtsuState())
    state = getattr(img_layer, key)
    return state

def _compute_iou(bin_mask1, bin_mask2):
    intersection = np.logical_and(bin_mask1, bin_mask2).sum()
    union = np.logical_or(bin_mask1, bin_mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def _remove_small_objects(bin_mask, min_size=50):
    labelled_mask = sk_label(bin_mask)
    for region in regionprops(labelled_mask):
        if region.area < min_size:
            labelled_mask[labelled_mask == region.label] = 0
    return (labelled_mask > 0).astype(np.uint8)


def _show_otsu_on_preview(threshold=None, use_watershed=False, state: OtsuState | None = None):
    """Perform Otsu segmentation within a 3D ROI and show it on Preview. 
    If the user draw a bounding box on slice z, the bounding box will be copied to all slices from z to form a 3D ROI for segmentation.
    The Otsu threshold will be calculated based on the image slices within the 3D ROI. 
    Pixel values >= threshold will be set as 1, otherwise 0.
    Then, it applies 
    1. a continuity check to the 3D Otsu mask, keeping only predictions with gaps (in the z direction) ≤ num_track_frame.
    2. an iou check, removing predictions that do not overlap with the previous 2 valid slices.
    3. post-processing by binary opening and closing.
    4. removing small components with size < 50 pixels.""" 

    viewer = napari.current_viewer()
    state = state or _get_otsu_state(viewer)
    # print("curr state", state)

    required_layers = ["Masks", "ROI", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    roi_layer = viewer.layers["ROI"]
    cd206_images = dataState.cd206_images

    if len(roi_layer.data) == 0 and state.last_bbox_xy is None:
        msg = "No bounding box"
        show_info(msg)
        print(msg)
        return  
    if len(roi_layer.data) != 0:
        last_bbox = roi_layer.data[-1][:,1:] 
        state.last_bbox_xy = last_bbox 
        y_min, x_min = last_bbox[0]
        y_max, x_max = last_bbox[2]
        # print(roi_layer.data[-1].shape) # (4, 3)
        # print(last_bbox) # [[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]]
        blocker = getattr(roi_layer.events.data, "blocker", None)
        if blocker is not None:
            with roi_layer.events.data.blocker():
                for z_idx in range(int(roi_layer.data[-1][0,0])+1, cd206_images.shape[0]):
                    curr_bbox = np.array([
                        [z_idx, y_min, x_min],
                        [z_idx, y_min, x_max],
                        [z_idx, y_max, x_max],
                        [z_idx, y_max, x_min]
                    ])
                    roi_layer.add(curr_bbox, shape_type="rectangle", edge_color="red", face_color="transparent", edge_width=2)

    if state.last_bbox_z is None:
        state.last_bbox_z = viewer.dims.current_step[0] # the slice index where the bbox is drawn
        # msg = f"Bounding box drawn on slice {state.last_bbox_z}"
        # show_info(msg)
        # print(msg)
    image_slice = cd206_images[state.last_bbox_z]
    bbox_mask = polygon2mask(image_slice.shape, state.last_bbox_xy) # shape (1024, 1024), binary
    
    # build 3D ROI volume
    start_frame = state.last_bbox_z
    num_frames = cd206_images.shape[0]-start_frame
    frames_to_segment = cd206_images[start_frame:]
    bbox_mask_3d = np.repeat(bbox_mask[np.newaxis, :, :], num_frames, axis=0)
    volume_in_bbox = frames_to_segment * bbox_mask_3d
    
    state.image_3d = volume_in_bbox
    state.bbox_mask_3d = bbox_mask_3d

    if threshold is None:
        threshold = threshold_otsu(volume_in_bbox[volume_in_bbox>0])
    state.threshold = threshold
    otsu_mask_3d = (volume_in_bbox >= threshold)
    # binary_mask_3d = (mask_layer.data[start_frame:] > 0)
    # otsu_mask_3d = otsu_mask_3d * (1-binary_mask_3d) # remove overlap with existing masks
    otsu_mask_3d = otsu_mask_3d.astype(np.uint8)
    # print("mask shape in preview", otsu_mask_3d.shape, use_watershed)
    # print("threshold in preview", threshold, use_watershed)

    mask_exist_slices = np.any(otsu_mask_3d, axis=(1, 2))
    mask_exist_slices = np.where(mask_exist_slices)[0]
    # print("111", mask_exist_slices+state.last_bbox_z)

    # continuity check
    num_track_frame = 3      
    valid_slices = [mask_exist_slices[0]]
    for prev, curr in zip(mask_exist_slices, mask_exist_slices[1:]):
        if curr-prev <= num_track_frame:
            valid_slices.append(curr)
        else:
            break
    # print("222", np.array(valid_slices)+state.last_bbox_z)

    # iou check
    iou_threshold = 0.0
    valid_slices_iou = [valid_slices[0]]
    for i in range(1, len(valid_slices)):
        curr = valid_slices[i]
        prev = valid_slices_iou[-1]
        prev2 = valid_slices_iou[-2] if len(valid_slices_iou) >= 2 else prev

        iou_prev = _compute_iou(otsu_mask_3d[curr], otsu_mask_3d[prev])
        iou_prev2 = _compute_iou(otsu_mask_3d[curr], otsu_mask_3d[prev2])
        if iou_prev > iou_threshold or iou_prev2 > iou_threshold:
            valid_slices_iou.append(curr)
        else:
            print(f"slice {curr+state.last_bbox_z} removed")
    # print("333", np.array(valid_slices_iou)+last_bbox_z)
        
    filtered_otsu_mask_3d = np.zeros_like(otsu_mask_3d, dtype=np.uint8)
    filtered_otsu_mask_3d[valid_slices_iou] = otsu_mask_3d[valid_slices_iou]
    
    # noise removal via binary opening and closing
    print("size before post-processing", filtered_otsu_mask_3d.sum())
    structure = np.ones((1,3,3), dtype=np.uint8)
    filtered_otsu_mask_3d = binary_opening(filtered_otsu_mask_3d, structure=structure)
    print("size after opening", filtered_otsu_mask_3d.sum())
    filtered_otsu_mask_3d = binary_closing(filtered_otsu_mask_3d, structure=structure)
    print("size after closing", filtered_otsu_mask_3d.sum())

    # remove small components
    filtered_otsu_mask_3d = _remove_small_objects(filtered_otsu_mask_3d, min_size=50)
    print("size after removing small components", filtered_otsu_mask_3d.sum())

    if use_watershed:
        distance = distance_transform_edt(filtered_otsu_mask_3d)
        full_bbox_size = (int(cd206_images.shape[1])) * (int(cd206_images.shape[2]))
        # print("full bbox size", full_bbox_size)
        # print("current bbox size", bbox_mask.sum())
        tol = 10
        if abs(bbox_mask.sum()-full_bbox_size) < tol:
            coords = peak_local_max(
                distance, 
                labels=filtered_otsu_mask_3d,
                footprint=np.ones((50, 50, 50), dtype=bool),
                num_peaks=100,
            ) 
            # print("num peaks", coords.shape[0])
        else:
            coords = peak_local_max(
                distance, 
                labels=filtered_otsu_mask_3d,
                num_peaks=1,
            ) # (n, 3)， n=2=num_peaks
            # print("num peaks", coords.shape[0])
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = label(mask)
        watershed_mask_3d = watershed(-distance, markers, mask=filtered_otsu_mask_3d)

    state.preview_mode = "watershed" if use_watershed else "otsu"
    msg = f"Showing {state.preview_mode} mask in Preview Mask layer"
    show_info(msg)

    if "Preview Mask" in viewer.layers:
        if not use_watershed:
            viewer.layers["Preview Mask"].data[state.last_bbox_z:] = filtered_otsu_mask_3d
        else:
            viewer.layers["Preview Mask"].data[state.last_bbox_z:] = watershed_mask_3d
    else:
        preview_data = np.zeros_like(viewer.layers["Masks"].data)
        if not use_watershed:
            preview_data[state.last_bbox_z:] = filtered_otsu_mask_3d
        else:
            preview_data[state.last_bbox_z:] = watershed_mask_3d
        viewer.add_labels(preview_data, name="Preview Mask")

    viewer.layers["Preview Mask"].refresh()


def _update_otsu_preview_by_slider(threshold: float=0.5):
    """Store threshold from slider and update Otsu preview mask"""
    viewer = napari.current_viewer()
    state = _get_otsu_state(viewer)
    if state.image_3d is None or state.bbox_mask_3d is None:
        return
    state.threshold = threshold
    _show_otsu_on_preview(threshold, use_watershed=False, state=state)


otsu_slider_widget = None
def _create_slider_or_update_otsu():
    global otsu_slider_widget
    need_new = otsu_slider_widget is None or getattr(otsu_slider_widget, "native", None) is None # A magicgui widget is a Python object that wraps a Qt widget internally. The underlying native Qt widget can be accessed through the .native attribute.
    if not need_new:
        try:
            _ = otsu_slider_widget.native.isVisible()
        except Exception:
            need_new = True
    if need_new:
        otsu_slider_widget = magicgui(
            _update_otsu_preview_by_slider,
            threshold={
                'widget_type': 'FloatSlider',
                'min': 0.0,
                'max': 255.0,
                'step': 1.0,
                'value': 0.0
            },
            call_button = "Rerun Otsu",
        )
    return otsu_slider_widget


def run_otsu_on_bbox():
    global otsu_slider_widget

    viewer = napari.current_viewer()
    required_layers = ["Masks", "ROI"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    state = _get_otsu_state(viewer)
    roi_layer = viewer.layers["ROI"]

    if len(roi_layer.data) > 0: 
        _show_otsu_on_preview(state = state)
        otsu_thresh = threshold_otsu(state.image_3d[state.image_3d>0])
        _create_slider_or_update_otsu()
        otsu_slider_widget.native.setEnabled(True)
        otsu_slider_widget.threshold.max = state.image_3d.max()
        otsu_slider_widget.threshold.value = otsu_thresh


def _run_watershed_on_whole_image():
    viewer = napari.current_viewer()
    state = _get_otsu_state(viewer)
    if state.image_3d is None or state.bbox_mask_3d is None:
        threshold = None
    else:
        threshold = state.threshold
    _show_otsu_on_preview(threshold=threshold, use_watershed=True, state=state)


def run_watershed_on_bbox():
    """Run watershed based on current Otsu preview and threshold"""
    viewer = napari.current_viewer()
    state = _get_otsu_state(viewer)
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    cd206_images = dataState.cd206_images
    has_roi = ("ROI" in viewer.layers) and (len(viewer.layers["ROI"].data) > 0)
    if not has_roi:
        reply = QMessageBox.question(
            None,
            "No bounding box detected",
            "No bounding box detected.\n"
            "Do you want to run watershed on the whole image? \n"
            "Press 'No' to draw a bounding box first (run Otsu) and then run watershed.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            # add_roi_layer(run_algo="otsu")
            return
        
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

        H, W = int(cd206_images.shape[1]), int(cd206_images.shape[2])
        state.last_bbox_z = 0
        full_bbox = np.array([
            [state.last_bbox_z, 0, 0],
            [state.last_bbox_z, 0, W-1],
            [state.last_bbox_z, H-1, W-1],
            [state.last_bbox_z, H-1, 0]
        ])
        # print("full bbox", full_bbox)
        blocker = getattr(roi_layer.events.data, "blocker", None)
        if blocker is not None:
            with roi_layer.events.data.blocker():
                roi_layer.add(full_bbox, shape_type="rectangle", edge_color="red", face_color="transparent", edge_width=2)
        else:
            roi_layer.add(full_bbox, shape_type="rectangle", edge_color="red", face_color="transparent", edge_width=2)
        _run_watershed_on_whole_image()
        otsu_thresh = threshold_otsu(state.image_3d[state.image_3d>0])
        otsu_slider_widget.threshold.value = otsu_thresh
        return

    _show_otsu_on_preview(threshold=state.threshold, use_watershed=True, state=state)


def finalise_mask(do_3d=True, use_watershed=False):
    viewer = napari.current_viewer()
    state = _get_otsu_state(viewer)
    required_layers = ["Masks", "ROI", "Preview Mask", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    cd206_images = dataState.cd206_images
    roi_layer = viewer.layers["ROI"]
    preview_layer = viewer.layers["Preview Mask"]
    max_object_id = viewer.layers["Masks"].data.max()

    if state.preview_mode == "watershed" and not use_watershed:
        msg = "Current mask is from watershed, do not save as Otsu"
        show_info(msg)
        print(msg)
        return
    if state.preview_mode == "otsu" and use_watershed:
        msg = "Current mask is from Otsu, do not save as Watershed"
        show_info(msg)
        print(msg)
        return

    image_slice = cd206_images[state.last_bbox_z]
    bbox_mask = polygon2mask(image_slice.shape, state.last_bbox_xy)
    otsu_mask_3d = preview_layer.data
    # print("Otsu Mask Shape in final:", otsu_mask_3d.shape)

    iou_threshold = 0.3
    exist = False
    similar_object_id = 0

    if not do_3d and not use_watershed:
        otsu_mask = otsu_mask_3d[state.last_bbox_z].astype(np.uint8)
        for z in [state.last_bbox_z+1, state.last_bbox_z+2]:
            if z >= cd206_images.shape[0]:
                break
            neighbour_mask_slice = viewer.layers["Masks"].data[z] * bbox_mask
            similar_object_id = np.unique(neighbour_mask_slice)
            similar_object_id = similar_object_id[similar_object_id != 0]
            intersection = np.logical_and(otsu_mask, neighbour_mask_slice).sum()
            union = np.logical_or(otsu_mask, neighbour_mask_slice).sum()
            if union > 0 and intersection / union > iou_threshold:
                exist = True
                break
        if exist:
            reply = QMessageBox.question(
                None,
                "Object already exists nearby",
                f"Object {similar_object_id[0]} already exists in a similar location in the following slices.\nDo you still want to create a new one?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                msg = "Did not create new object"
                show_info(msg)
                print(msg)
                viewer.layers.remove(roi_layer)
                viewer.layers.remove(viewer.layers["Preview Mask"])
                viewer.layers.selection.add(viewer.layers["Masks"])
                return

        if otsu_mask.sum() > 0:
            mask_data = viewer.layers["Masks"].data.copy()
            mask_data[state.last_bbox_z][otsu_mask==1] = max_object_id+1
            viewer.layers["Masks"].data = mask_data

            viewer.layers.remove(roi_layer)
            viewer.layers.remove(viewer.layers["Preview Mask"])
            viewer.layers["Masks"].refresh()
            viewer.layers.selection.add(viewer.layers["Masks"])

            msg = f"Added new object {max_object_id+1} at slice {state.last_bbox_z} using Otsu's method"
            show_info(msg)
            print(msg)   
        else:
            msg = f"The object created overlap completely with an existing object, did not create new object"
            show_info(msg)
            print(msg)
            viewer.layers.remove(roi_layer)
            viewer.layers.remove(viewer.layers["Preview Mask"])
            viewer.layers["Masks"].refresh()
            viewer.layers.selection.add(viewer.layers["Masks"])

    else:
        otsu_mask_3d = otsu_mask_3d.astype(np.uint8)
        mask_exist_slices = np.any(otsu_mask_3d, axis=(1, 2))
        mask_exist_slices = np.where(mask_exist_slices)[0]
        if len(mask_exist_slices) == 0:
            show_info("No mask to add")
            viewer.layers.remove(roi_layer)
            viewer.layers.remove(viewer.layers["Preview Mask"]) 
            viewer.layers.selection.add(viewer.layers["Masks"])
            return
        for z in [state.last_bbox_z+1, mask_exist_slices[-1]+2]:
            if z >= cd206_images.shape[0]:
                break
            neighbour_mask_slice = viewer.layers["Masks"].data[z] * bbox_mask
            similar_object_id = np.unique(neighbour_mask_slice)
            similar_object_id = similar_object_id[similar_object_id != 0]
            intersection = np.logical_and(otsu_mask_3d[z], neighbour_mask_slice).sum()
            union = np.logical_or(otsu_mask_3d[z], neighbour_mask_slice).sum()
            if union > 0 and intersection / union > iou_threshold:
                exist = True
                break

        if exist:
            reply = QMessageBox.question(
                None,
                "Object already exists nearby",
                f"Object {similar_object_id[0]} already exists in a similar location in the following slices.\nDo you still want to create a new one?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                msg = "Did not create new object"
                show_info(msg)
                print(msg)
                viewer.layers.remove(roi_layer)
                viewer.layers.remove(viewer.layers["Preview Mask"])
                viewer.layers.selection.add(viewer.layers["Masks"])
                return

        if otsu_mask_3d.sum() > 0:
            mask_data = viewer.layers["Masks"].data.copy()
            if not use_watershed:
                mask_data[otsu_mask_3d==1] = max_object_id+1
                msg = f"Added new object {max_object_id+1} on slices {mask_exist_slices} using Otsu's method"
                show_info(msg)
                print(msg) 
            else:
                watershed_mask_3d = otsu_mask_3d.copy()
                watershed_mask_3d[watershed_mask_3d > 0] += max_object_id
                mask_data[watershed_mask_3d > 0] = watershed_mask_3d[watershed_mask_3d > 0]
                msg = f"Added new objects {max_object_id+1}-{watershed_mask_3d.max()} on slices {mask_exist_slices} using Watershed"
                show_info(msg)
                print(msg)

            viewer.layers["Masks"].data = mask_data
            viewer.layers.remove(roi_layer)
            viewer.layers.remove(viewer.layers["Preview Mask"])
            viewer.layers["Masks"].refresh()
            viewer.layers.selection.add(viewer.layers["Masks"])

        else:
            msg = f"The object created overlap completely with an existing object, did not create new object"
            show_info(msg)
            print(msg)
            viewer.layers.remove(roi_layer)
            viewer.layers.remove(viewer.layers["Preview Mask"])
            viewer.layers["Masks"].refresh()
            viewer.layers.selection.add(viewer.layers["Masks"])

    state.last_bbox_z = None
    state.last_bbox_xy = None


##### Run watershed segmentation on all ROIs #####
def run_watershed_for_all_rois():
    """Run Watershed inside every ROI rectangle and write results directly into Masks layer with new object IDs."""

    viewer = napari.current_viewer()
    cd206_images = dataState.cd206_images
    if "CD206" not in viewer.layers:
        if cd206_images is None:
            msg = "Image not loaded"
            show_warning(msg)
            return
        else:
            viewer.add_image(cd206_images, name="CD206", blending="additive")
    if "ROI" not in viewer.layers:
        _layer_not_in_viewer_error("ROI")
        return  
    if "Masks" not in viewer.layers:
        mask_layer = viewer.add_labels(np.zeros_like(cd206_images, dtype=np.uint8), name="Masks")
        if select_object not in getattr(mask_layer, "mouse_drag_callbacks", []):
            mask_layer.mouse_drag_callbacks.append(select_object)  # select_object will be called when user drags mouse
        mask_layer.selected_object_id = None
        mask_layer.click_coords = None

    roi_layer = viewer.layers["ROI"]
    mask_layer = viewer.layers["Masks"]

    mask_data = mask_layer.data.copy()
    next_id = int(mask_data.max()) + 1
    total_added = 0

    for curr_box in roi_layer.data:
        curr_slice_idx = int(curr_box[0, 0])
        bbox_corners_yx = curr_box[:, 1:]  # [[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]]

        # build 3d ROI
        image_slice = cd206_images[curr_slice_idx]
        bbox_mask_2d = polygon2mask(image_slice.shape, bbox_corners_yx).astype(np.uint8)
        frames_to_segment = cd206_images[curr_slice_idx:]
        bbox_mask_3d = np.repeat(bbox_mask_2d[np.newaxis, ...], frames_to_segment.shape[0], axis=0)
        vol_in_bbox = frames_to_segment * bbox_mask_3d

        if np.count_nonzero(vol_in_bbox) == 0:
            continue

        threshold = threshold_otsu(vol_in_bbox[vol_in_bbox > 0])
        otsu_mask_3d = (vol_in_bbox >= threshold).astype(np.uint8)

        # continuity check
        exist_slices = np.where(np.any(otsu_mask_3d, axis=(1, 2)))[0]
        if exist_slices.size == 0:
            continue
        num_track_frame = 3
        valid_slices = [exist_slices[0]]
        for prev, curr in zip(exist_slices, exist_slices[1:]):
            if curr - prev <= num_track_frame:
                valid_slices.append(curr)
            else:
                break

        # IoU check
        iou_threshold = 0.0
        valid_slices_iou = [valid_slices[0]]
        for i in range(1, len(valid_slices)):
            curr = valid_slices[i]
            prev = valid_slices_iou[-1]
            prev2 = valid_slices_iou[-2] if len(valid_slices_iou) >= 2 else prev
            if _compute_iou(otsu_mask_3d[curr], otsu_mask_3d[prev]) > iou_threshold or _compute_iou(otsu_mask_3d[curr], otsu_mask_3d[prev2]) > iou_threshold:
                valid_slices_iou.append(curr)

        filtered = np.zeros_like(otsu_mask_3d, dtype=np.uint8)
        filtered[valid_slices_iou] = otsu_mask_3d[valid_slices_iou]

        # post-processing
        structure = np.ones((1, 3, 3), dtype=np.uint8)
        filtered = binary_opening(filtered, structure=structure)
        filtered = binary_closing(filtered, structure=structure)
        filtered = _remove_small_objects(filtered, min_size=50)

        if np.count_nonzero(filtered) == 0:
            continue

        # Watershed
        distance = distance_transform_edt(filtered)
        coords = peak_local_max(distance, labels=filtered, num_peaks=1)
        peak_mask = np.zeros(distance.shape, dtype=bool)
        if coords.size > 0:
            peak_mask[tuple(coords.T)] = True
        markers, _ = label(peak_mask)
        ws = watershed(-distance, markers, mask=filtered)

        if np.any(ws):
            for lbl in np.unique(ws):
                if lbl == 0:
                    continue
                binmask = (ws == lbl)
                mask_data_slice = mask_data[curr_slice_idx:]
                mask_data_slice[binmask] = next_id
                mask_data[curr_slice_idx:] = mask_data_slice
                next_id += 1
                total_added += 1

    if total_added > 0:
        mask_layer.data = mask_data
        mask_layer.refresh()
        show_info(f"Added {total_added} object(s) using Watershed in {len(roi_layer.data)} ROI(s)")
    else:
        show_info("No objects added from ROI watershed")
