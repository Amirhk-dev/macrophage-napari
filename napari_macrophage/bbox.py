import os
import napari
import numpy as np
import re

from pathlib import Path
from napari.layers import Labels as _Labels
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QFileDialog, QMessageBox

from .state import dataState
from .error import _layers_not_in_viewer_error
from .segmentation import run_otsu_on_bbox


def add_roi_layer(run_algo=None):
    """ Add ROI layer. Force the layer to store bb of shape (4, 3). 
    If run_algo is "otsu", connect the bbox drawing to run_otsu_on_bbox."""
    viewer = napari.current_viewer()
    required_layers = ["CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return

    if "DAPI" in viewer.layers:
        viewer.layers["DAPI"].visible = False

    if "Preview Mask" in viewer.layers:
        msg = "Please save any preview mask"
        show_warning(msg)
        return
    
    if "ROI" not in viewer.layers:
        roi_layer = viewer.add_shapes(name="ROI", shape_type="rectangle", edge_color="white", face_color="transparent", edge_width=3)
        curr_z = int(viewer.dims.current_step[0])
        _dummy = np.array([ # to force the layer to store bb of shape (4, 3) 
            [curr_z, 0, 0],
            [curr_z, 0, 0],
            [curr_z, 0, 0],
            [curr_z, 0, 0],
        ])
        idx = len(roi_layer.data)
        roi_layer.add(_dummy, shape_type="rectangle", edge_color="white", face_color="transparent", edge_width=3)
        roi_layer.selected_data = {idx}
        roi_layer.remove_selected()
        if curr_z < viewer.layers["CD206"].data.shape[0] - 1: # simulate a change of slice to force refresh
            viewer.dims.set_current_step(0, curr_z+1)
        else:
            viewer.dims.set_current_step(0, curr_z-1)
        viewer.dims.set_current_step(0, curr_z)
    else:
        roi_layer = viewer.layers["ROI"]
        viewer.layers.selection.add(roi_layer)
        if viewer.layers["Masks"] in viewer.layers.selection:
            viewer.layers.selection.remove(viewer.layers["Masks"])

    msg = "Please draw a bounding box"
    show_info(msg)
    print(msg)

    if run_algo == "otsu":
        if len(roi_layer.data) == 0:
            roi_layer.events.data.connect(run_otsu_on_bbox)
    # elif run_algo is None and len(roi_layer.data) == 0:
    #     roi_layer.events.data.connect(draw_bboxes)


###### import or export COCO-style JSON/ YOLO-style txt ######
def draw_bboxes(*args, **kwargs):
    viewer = napari.current_viewer()
    roi_layer = viewer.layers["ROI"]
    # print("111", roi_layer.data)
    if len(roi_layer.data) == 0:
        msg = "No bounding box"
        return
    else:
        try:
            last_box = roi_layer.data[-1]
        except Exception:
            show_warning("Failed to access last ROI")
            return
        ys = last_box[:, 1]
        xs = last_box[:, 2]
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        # y_min, x_min = last_box[0][1:3] # this would disable drawing bb from other directions, because x_max - x_min may be negative and truncated to zero in w later
        # y_max, x_max = last_box[2][1:3]
        curr_slice = viewer.dims.current_step[0]
        msg = f"Bbox at {int(y_min), int(x_min)} to {int(y_max), int(x_max)} on slice {curr_slice} changed"
        show_info(msg)
        # print(msg)


def generate_bboxes_from_mask_layer():
    """Generate per-slice bounding boxes for every label > 0 in the selected Labels layer (falls back to 'Masks')."""
    viewer = napari.current_viewer()

    active = viewer.layers.selection.active
    if isinstance(active, _Labels):
        mask_layer = active
    elif "Masks" in viewer.layers:
        mask_layer = viewer.layers["Masks"]
    else:
        show_warning("Select a Labels layer or load a Masks layer first.")
        return

    data = np.asarray(mask_layer.data)
    if data.ndim != 3:
        show_warning("Mask layer must be 3D (Z, Y, X)")
        return

    if "ROI" not in viewer.layers:
        roi_layer = viewer.add_shapes(name="ROI", shape_type="rectangle", edge_color="white", face_color="transparent", edge_width=2)
    else:
        roi_layer = viewer.layers["ROI"]
        reply = QMessageBox.question(
            None,
            "Clear existing ROIs?",
            "Clear existing ROIs before generating from mask?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                viewer.layers.remove(roi_layer)
            except Exception:
                pass
            roi_layer = viewer.add_shapes(name="ROI", shape_type="rectangle", edge_color="white", face_color="transparent", edge_width=2)
    
    Z = int(data.shape[0])
    added = 0

    blocker = getattr(roi_layer.events.data, "blocker", None)
    if blocker is not None:
        with roi_layer.events.data.blocker():
            for z in range(Z):
                slice2d = data[z]
                labels = np.unique(slice2d)
                labels = labels[labels != 0]
                for lab in labels:
                    ys, xs = np.where(slice2d == lab)
                    if ys.size == 0:
                        continue
                    y0, y1 = int(ys.min()), int(ys.max()) 
                    x0, x1 = int(xs.min()), int(xs.max())
                    if x1 <= x0 or y1 <= y0:
                        continue
                    curr_bbox = np.array([
                        [z, y0, x0],
                        [z, y0, x1],
                        [z, y1, x1],
                        [z, y1, x0],
                    ], dtype=float)
                    roi_layer.add(
                        curr_bbox, 
                        shape_type="rectangle", 
                        edge_color="white", 
                        face_color="transparent", 
                        edge_width=2
                    )
                    added += 1
    show_info(f"Generated {added} bounding boxes from {mask_layer.name}")
    print(f"Generated {added} bounding boxes from {mask_layer.name}")


def export_bboxes_to_yolo():
    """Export ROIs to YOLO txt files. One txt per slice: {image_name}_slice_{z}.txt"""

    viewer = napari.current_viewer()
    required_layers = ["ROI", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    file_name = dataState.file_name
    cd206_images = dataState.cd206_images
    roi_layer = viewer.layers["ROI"]
    if len(roi_layer.data) == 0:
        show_warning("No bounding box to export")
        return

    out_dir = QFileDialog.getExistingDirectory(
        None, "Select Output Folder for YOLO labels", ""
    )
    if not out_dir:
        return

    if cd206_images is not None:
        H, W = int(cd206_images.shape[1]), int(cd206_images.shape[2])
    else:
        show_warning("Cannot determine image size for normalisation")
        return
    if W <= 0 or H <= 0:
        show_warning("Invalid image size")
        return

    # group bboxes by slice
    per_slice = {}
    for curr_box in roi_layer.data:
        z_idx = int(curr_box[0][0])
        ys = curr_box[:, 1]
        xs = curr_box[:, 2]
        y_min = max(0.0, min(np.float32(H - 1), ys.min()))
        y_max = max(0.0, min(np.float32(H - 1), ys.max()))
        x_min = max(0.0, min(np.float32(W - 1), xs.min()))
        x_max = max(0.0, min(np.float32(W - 1), xs.max()))
        # y_min, x_min = curr_box[0][1:3]
        # y_min = max(0.0, min(np.float32(H - 1), y_min))
        # x_min = max(0.0, min(np.float32(W - 1), x_min))
        # y_max, x_max = curr_box[2][1:3]
        # y_max = max(0.0, min(np.float32(H - 1), y_max))
        # x_max = max(0.0, min(np.float32(W - 1), x_max))

        w = max(0, x_max-x_min)
        h = max(0, y_max-y_min)
        if w == 0 or h == 0:
            continue

        x_center = (x_max+x_min) / 2.0
        y_center = (y_max+y_min) / 2.0

        x_c_n = np.float32(x_center) / np.float32(W)
        y_c_n = np.float32(y_center) / np.float32(H)
        w_n = np.float32(w) / np.float32(W)
        h_n = np.float32(h) / np.float32(H)

        per_slice.setdefault(z_idx, []).append((x_c_n, y_c_n, w_n, h_n))

    if not per_slice:
        show_info("No valid bounding boxes to export")
        return

    files_written = 0
    boxes_written = 0
    # print("per_slice", per_slice)

    for z_idx, rows in per_slice.items():
        txt_name = f"{file_name}_slice_{z_idx}.txt"
        out_path = os.path.join(out_dir, txt_name)
        with open(out_path, "w") as f:
            for cls, (xc, yc, ww, hh) in enumerate(rows):
                cls = 0  # one class only
                f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
                boxes_written += 1
        files_written += 1

    msg = f"Exported {files_written} YOLO files, {boxes_written} boxes to {out_dir}"
    show_info(msg)
    print(msg)


def import_bboxes_from_yolo_folder():
    """Import YOLO txt labels from a folder. Only files that match {file_name}_slice_{z}.txt are loaded."""

    viewer = napari.current_viewer()
    file_name = dataState.file_name 
    cd206_images = dataState.cd206_images
    if not file_name:
        show_warning("Please load image first before loading its bounding boxes.")
        return

    if cd206_images is not None:
        H, W = int(cd206_images.shape[1]), int(cd206_images.shape[2])
    else:
        show_warning("Cannot determine image size. Load image first.")
        return

    dir_path = QFileDialog.getExistingDirectory(
        None, "Select Folder with YOLO labels", ""
    )
    if not dir_path:
        return

    if "ROI" not in viewer.layers:
        roi_layer = viewer.add_shapes(
            name="ROI",
            shape_type="rectangle",
            edge_color="white",
            face_color="transparent",
            edge_width=2
        )
    else:
        roi_layer = viewer.layers["ROI"]

    patt = re.compile(rf"^{re.escape(file_name)}_slice_(\d+)\.txt$")
    txt_files = [f for f in os.listdir(dir_path) if patt.match(f)]
    if not txt_files:
        show_info("No matching YOLO files for current image")
        return

    total_files = 0
    total_boxes = 0

    blocker = getattr(roi_layer.events.data, "blocker", None)
    if blocker is not None:
        with roi_layer.events.data.blocker():
            for fname in sorted(txt_files):
                m = patt.match(fname)
                if not m:
                    continue
                z_idx = int(m.group(1))
                fpath = os.path.join(dir_path, fname)
    
                try:
                    with open(fpath, "r") as fh:
                        lines = fh.readlines()
                except Exception:
                    continue

                added_this_file = 0
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                       continue
                    try:
                        _cls = int(parts[0])
                        xc_n = np.float32(parts[1])
                        yc_n = np.float32(parts[2])
                        w_n = np.float32(parts[3])
                        h_n = np.float32(parts[4])
                    except ValueError:
                        continue

                    w = w_n * W
                    h = h_n * H
                    if w <= 0 or h <= 0:
                        continue
                    xc = xc_n * W
                    yc = yc_n * H
                    x_min = xc - w / 2.0
                    y_min = yc - h / 2.0
                    x_max = xc + w / 2.0
                    y_max = yc + h / 2.0

                    x_min = max(0.0, min(np.float32(W - 1), x_min))
                    y_min = max(0.0, min(np.float32(H - 1), y_min))
                    x_max = max(0.0, min(np.float32(W - 1), x_max))
                    y_max = max(0.0, min(np.float32(H - 1), y_max))
                    if x_max <= x_min or y_max <= y_min:
                        continue

                    curr_bbox = np.array([
                        [z_idx, y_min, x_min],
                        [z_idx, y_min, x_max],
                        [z_idx, y_max, x_max],
                        [z_idx, y_max, x_min],
                   ], dtype=float)

                    roi_layer.add(
                        curr_bbox,
                        shape_type="rectangle",
                        edge_color="white",
                        face_color="transparent",
                        edge_width=2
                    )
                    total_boxes += 1
                    added_this_file += 1

                if added_this_file > 0:
                    total_files += 1
    
    curr_z = int(viewer.dims.current_step[0])
    if curr_z < viewer.layers["CD206"].data.shape[0] - 1: # simulate a change of slice to force refresh
        viewer.dims.set_current_step(0, curr_z+1)
    else:
        viewer.dims.set_current_step(0, curr_z-1)
    viewer.dims.set_current_step(0, curr_z)
    
    msg = f"Imported {total_boxes} boxes from {total_files} files in {dir_path}"
    show_info(msg)
    print(msg)


def detect_objects_with_onnx(
    onnx_path: Path = None,
    confidence_threshold: float = 0.5,
    current_slice_only: bool = False,
):
    """Run ONNX object detection using CD206 (R), zeros (G), DAPI (B) → input shape (1,3,H,W).
    Detected boxes are added to the ROI shapes layer in the same format as manually drawn boxes."""
    try:
        import onnxruntime as ort
    except ImportError:
        show_warning("onnxruntime is not installed. Run: pip install onnxruntime")
        return

    viewer = napari.current_viewer()
    cd206 = dataState.cd206_images
    dapi  = dataState.dapi_images

    if cd206 is None or dapi is None:
        show_warning("Both CD206 and DAPI channels must be loaded before running detection.")
        return

    if onnx_path is None or not Path(onnx_path).exists():
        show_warning("Please select a valid ONNX model file.")
        return

    sess       = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_meta   = sess.get_inputs()[0]
    out_meta   = sess.get_outputs()[0]
    input_name = inp_meta.name
    print(f"[ONNX] input  : name={inp_meta.name}  shape={inp_meta.shape}  type={inp_meta.type}")
    print(f"[ONNX] output : name={out_meta.name}  shape={out_meta.shape}")

    Z, H, W = cd206.shape
    print(f"[ONNX] image  : Z={Z}  H={H}  W={W}")

    # Normalise channels globally to [0, 1]
    cd206_norm = cd206.astype(np.float32) / (cd206.max() or 1)
    dapi_norm  = dapi.astype(np.float32)  / (dapi.max()  or 1)
    zeros      = np.zeros((H, W), dtype=np.float32)

    z_range = [int(viewer.dims.current_step[0])] if current_slice_only else range(Z)

    if "ROI" not in viewer.layers:
        roi_layer = viewer.add_shapes(
            name="ROI", shape_type="rectangle",
            edge_color="white", face_color="transparent", edge_width=2
        )
    else:
        roi_layer = viewer.layers["ROI"]

    total_boxes = 0
    blocker = getattr(roi_layer.events.data, "blocker", None)

    def _run():
        nonlocal total_boxes
        for i, z in enumerate(z_range):
            # Input: R=CD206, G=zeros, B=DAPI → (1, 3, H, W)
            inp = np.stack([cd206_norm[z], zeros, dapi_norm[z]], axis=0)[np.newaxis]
            raw = sess.run(None, {input_name: inp})[0]

            if i == 0:
                print(f"[ONNX] raw output shape : {raw.shape}")

            # Normalise output to shape (N, >=5)
            preds = np.squeeze(raw)
            if preds.ndim == 1:
                preds = preds[np.newaxis]
            if preds.ndim == 2 and preds.shape[0] == 5:
                preds = preds.T  # (5, N) → (N, 5)

            if i == 0:
                scores = preds[:, 4] if preds.ndim == 2 and preds.shape[1] >= 5 else np.array([])
                print(f"[ONNX] detections       : {len(preds)}")
                if scores.size:
                    print(f"[ONNX] score range      : {scores.min():.4f} – {scores.max():.4f}  (threshold={confidence_threshold})")
                    print(f"[ONNX] sample coords (first 3): {preds[:3, :4].tolist()}")

            for det in preds:
                if len(det) < 5:
                    continue
                xc, yc, bw, bh, score = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
                if score < confidence_threshold:
                    continue
                x_min = max(0.0, xc - bw / 2)
                x_max = min(float(W - 1), xc + bw / 2)
                y_min = max(0.0, yc - bh / 2)
                y_max = min(float(H - 1), yc + bh / 2)
                if x_max <= x_min or y_max <= y_min:
                    continue
                rect = np.array([
                    [z, y_min, x_min],
                    [z, y_min, x_max],
                    [z, y_max, x_max],
                    [z, y_max, x_min],
                ], dtype=float)
                roi_layer.add(rect, shape_type="rectangle",
                              edge_color="white", face_color="transparent", edge_width=2)
                total_boxes += 1

    if blocker is not None:
        with roi_layer.events.data.blocker():
            _run()
    else:
        _run()

    # Refresh slice view
    curr_z = int(viewer.dims.current_step[0])
    step   = curr_z + 1 if curr_z < Z - 1 else curr_z - 1
    viewer.dims.set_current_step(0, step)
    viewer.dims.set_current_step(0, curr_z)

    scope = "current slice" if current_slice_only else f"{Z} slices"
    show_info(f"Detection complete: {total_boxes} boxes across {scope}.")