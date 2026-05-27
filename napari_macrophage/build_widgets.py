import napari
import numpy as np

from magicgui import magicgui
from qtpy import QtWidgets, QtCore
from napari.utils.notifications import show_info, show_warning

from .io import add_image_layer, add_mask_layer, add_layer_from_zarr
from .edit_mask_image import delete_all, delete_object, edit_object_id, renumber, add_object_layer, sync_object_to_masks, _step_object_in_slice, select_object, interpolate_to_isotropic, shrink_mask_to_cd206
from .segmentation import run_watershed_for_all_rois, finalise_mask, run_watershed_on_bbox, _create_slider_or_update_otsu, _create_slider_or_update_otsu
from .bbox import add_roi_layer, generate_bboxes_from_mask_layer, export_bboxes_to_yolo, import_bboxes_from_yolo_folder, detect_objects_with_onnx
from .analysis import cells_analysis
from .ui import _widget_stylesheet, _set_call_button_tooltip
from .state import dataState, set_voxel_size_um
from .io import _prepare_all_layers


def _register_keyboard_shortcuts(viewer):
    viewer.bind_key("d", delete_object, overwrite=True)
    viewer.bind_key("Shift+D", delete_all, overwrite=True)
    viewer.bind_key("v", add_object_layer, overwrite=True)
    viewer.bind_key("i", cells_analysis, overwrite=True) # bind_key will pass viewer as first argument
    viewer.bind_key("Up",  lambda v: _step_object_in_slice(-1), overwrite=True)
    viewer.bind_key("Down", lambda v: _step_object_in_slice(+1), overwrite=True)


main_tools_dock = None
def _built_widgets():
    """Create a single scrollable 'Macrophage Tools' dock that contains all tool groups."""

    viewer = napari.current_viewer()
    local_style = _widget_stylesheet()

    global main_tools_dock
    try:
        if main_tools_dock is not None and main_tools_dock.isVisible():
            return
    except RuntimeError:
        main_tools_dock = None

    delete_this_widget = magicgui(
        delete_object, 
        call_button="Delete in Slice"
    )
    delete_all_widget = magicgui(
        delete_all, 
        call_button="Delete in ALL Slice"
    )

    change_id_widget = magicgui(
        edit_object_id, 
        new_id={"label": "New ID", "min": 1, "step": 1}, 
        call_button="Change ID"
    )
    for spin in change_id_widget.native.findChildren(QtWidgets.QSpinBox):
        spin.setStyleSheet("font-size: 10pt;")
    new_id_widget = magicgui(
        lambda: edit_object_id(new_id=None),
        call_button="Auto Assign ID"
    )

    view_object_widget = magicgui(
        add_object_layer,
        object_id={"label": "Object ID", "min": 1, "step": 1},
        call_button="View Object"
    )
    apply_changes_widget = magicgui(
        sync_object_to_masks,
        call_button="Apply Changes"
    )
    shrink_mask_widget = magicgui(
        shrink_mask_to_cd206,
        call_button="Shrink Mask"
    )

    add_roi_widget = magicgui(
        lambda: add_roi_layer(run_algo="otsu"), 
        call_button="Add BBox"
    )
    finalise_3d_widget = magicgui(
        lambda: finalise_mask(do_3d=True, use_watershed=False), 
        call_button="Save Otsu 3D"
    )
    watershed_widget = magicgui(
        run_watershed_on_bbox, 
        call_button="Run Watershed"
    )
    finalise_watershed_widget = magicgui(
        lambda: finalise_mask(do_3d=True, use_watershed=True), 
        call_button="Save Watershed 3D"
    )
    otsu_slider_widget = _create_slider_or_update_otsu()
    otsu_slider_widget.native.setStyleSheet(_widget_stylesheet())
    slider_native = otsu_slider_widget.native

    bboxes_widget = magicgui(
        add_roi_layer, 
        call_button="Draw BBox"
    )
    gen_bboxes_widget = magicgui(
        generate_bboxes_from_mask_layer, 
        call_button="Generate All BBox"
    )

    export_yolo_widget = magicgui(
        export_bboxes_to_yolo,
        call_button="Export BBs (YOLO)"
    )
    import_yolo_widget = magicgui(
        import_bboxes_from_yolo_folder,
        call_button="Import BBs (YOLO)"
    )
    detect_onnx_widget = magicgui(
        detect_objects_with_onnx,
        call_button="Detect BBs (ONNX)",
        onnx_path={"label": "ONNX model (.onnx)", "filter": "*.onnx"},
        confidence_threshold={"label": "Confidence", "min": 0.0, "max": 1.0, "step": 0.05},
        current_slice_only={"label": "Current slice only"},
    )
    for le in detect_onnx_widget.onnx_path.native.findChildren(QtWidgets.QLineEdit):
        le.setReadOnly(True)
        le.setText("")
        le.setPlaceholderText("Default model (select to override)")

    image_info_widget = magicgui(
        set_voxel_size_um, 
        voxel_x={"label": "Voxel size X [µm]"},
        voxel_y={"label": "Voxel size Y [µm]"},
        voxel_z={"label": "Voxel size Z [µm]"},
        call_button="Update Voxel Size"
    )
    cells_analysis_widget = magicgui(
        cells_analysis, 
        call_button="Cells Analysis"
    )
    interpolate_widget = magicgui(
        interpolate_to_isotropic, 
        call_button="Interpolate to Isotropic"
    )
    renumber_widget = magicgui(
        renumber, 
        call_button="Renumber All"
    )


    _set_call_button_tooltip(delete_this_widget, "[d] Delete the selected connected component in the current slice.")
    _set_call_button_tooltip(delete_all_widget, "[Shift+d] Delete the selected object completely throughout the image.")

    _set_call_button_tooltip(change_id_widget, "Relabel the selected component to the given ID.")
    _set_call_button_tooltip(new_id_widget, "Assign the next available ID to the selected component.")

    _set_call_button_tooltip(view_object_widget, "[v] View the selected object only in a new layer with name Object {ID}. You can either click on the object to select it or enter the ID manually. The click has a higher priority.")
    _set_call_button_tooltip(apply_changes_widget, "Save changes made on the Object layer back to the Masks layer.")
    _set_call_button_tooltip(shrink_mask_widget, "Shrink the selected object's mask to fit the real CD206 boundaries. Uses morphological Chan-Vese (region-based active contour) initialised from the current mask — pulls inward where CD206 signal is weak.")

    _set_call_button_tooltip(add_roi_widget, "Draw a bounding box on the ROI layer. 3D Otsu segmentation will be automatically applied to the last drawn box, and the result will appear in the Preview Mask layer. You can adjust the threshold using the slider below and rerun Otsu if needed.")
    _set_call_button_tooltip(finalise_3d_widget, "Save the current Otsu segmentation result back to Mask layer.")
    _set_call_button_tooltip(watershed_widget, "Run Watershed based on the current Otsu preview or on the whole image if no bounding box is detected.")
    _set_call_button_tooltip(finalise_watershed_widget, "Save the current Watershed segmentation result back to Mask layer.")

    _set_call_button_tooltip(bboxes_widget, "Draw bounding boxes on the ROI layer. If you want to continue annotating from an existing file, please import the file first before drawing new bounding boxes.")
    _set_call_button_tooltip(gen_bboxes_widget, "Generate bounding boxes from the Masks layer across all slices.")
    _set_call_button_tooltip(export_yolo_widget, "Export the current bounding boxes to YOLO txt files (one file per slice).")
    _set_call_button_tooltip(import_yolo_widget, "Import YOLO txt files from a folder.")
    _set_call_button_tooltip(detect_onnx_widget, "Run ONNX object detection on all Z slices using CD206 + DAPI (CPU). Results are added to the ROI layer and can be edited or exported like any other bounding box.")
    
    _set_call_button_tooltip(image_info_widget, "Update the voxel size information used for analysis and processing.")
    _set_call_button_tooltip(cells_analysis_widget, "Compute volume and sphericity for each cell based on Masks layer.")
    _set_call_button_tooltip(interpolate_widget, "Interpolate the current image/mask layer to isotropic voxel size.")
    _set_call_button_tooltip(renumber_widget, "Renumber all objects to consecutive IDs by their first appearance in the image.")

    def _row(*widgets: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout: # set horizontal layout 
        r = QtWidgets.QHBoxLayout()
        r.setSpacing(4)
        for w in widgets:
            layout = w.layout()
            if layout is not None:
                layout.setContentsMargins(4, 4, 4, 4)
                layout.setSpacing(4)
            r.addWidget(w)
        return r

    object_group = QtWidgets.QGroupBox("Objects")
    object_v = QtWidgets.QVBoxLayout()
    object_v.setContentsMargins(4, 4, 4, 4)
    object_v.setSpacing(6)
    object_v.addLayout(_row(delete_this_widget.native, delete_all_widget.native))
    object_v.addLayout(_row(change_id_widget.native, new_id_widget.native))
    object_v.addLayout(_row(view_object_widget.native, apply_changes_widget.native))
    object_v.addLayout(_row(shrink_mask_widget.native))
    object_group.setLayout(object_v)

    seg_group = QtWidgets.QGroupBox("Segmentation")
    seg_v = QtWidgets.QVBoxLayout()
    seg_v.setContentsMargins(4, 4, 4, 4)
    seg_v.setSpacing(4)
    seg_v.addLayout(_row(add_roi_widget.native, finalise_3d_widget.native))
    seg_v.addLayout(_row(watershed_widget.native, finalise_watershed_widget.native))
    seg_v.addLayout(_row(slider_native))
    seg_group.setLayout(seg_v)

    bbox_group = QtWidgets.QGroupBox("Bounding Boxes (BBs)")
    bbox_v = QtWidgets.QVBoxLayout()
    bbox_v.setContentsMargins(4, 4, 4, 4)
    bbox_v.setSpacing(4)
    bbox_v.addLayout(_row(bboxes_widget.native, gen_bboxes_widget.native))
    bbox_v.addLayout(_row(export_yolo_widget.native, import_yolo_widget.native))
    bbox_v.addLayout(_row(detect_onnx_widget.native))
    bbox_group.setLayout(bbox_v)

    voxel_group = QtWidgets.QGroupBox("Voxel Size")
    voxel_v = QtWidgets.QVBoxLayout()
    voxel_v.setContentsMargins(4, 4, 4, 4)
    voxel_v.setSpacing(4)
    voxel_v.addWidget(image_info_widget.native)
    voxel_group.setLayout(voxel_v)

    analysis_processing_group = QtWidgets.QGroupBox("Analysis and Processing")
    ap_v = QtWidgets.QVBoxLayout()
    ap_v.setContentsMargins(4, 4, 4, 4)
    ap_v.setSpacing(4)
    for w in [renumber_widget.native, cells_analysis_widget.native, interpolate_widget.native]:
        layout = w.layout()
        if layout is not None:
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
        ap_v.addWidget(w)
    analysis_processing_group.setLayout(ap_v)

    # scrollable root container
    content = QtWidgets.QWidget()
    root = QtWidgets.QVBoxLayout(content)
    root.setContentsMargins(4, 4, 4, 4)
    root.setSpacing(4)
    for g in (object_group, seg_group, bbox_group, voxel_group, analysis_processing_group):
        g.setStyleSheet(local_style)
        root.addWidget(g)
    root.addStretch(1)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    wrapper = QtWidgets.QWidget()
    wl = QtWidgets.QVBoxLayout(wrapper)
    wl.setContentsMargins(0, 0, 0, 0)
    wl.setSpacing(0)
    wl.addWidget(scroll)
    wrapper.setStyleSheet(local_style)

    curr_dock = viewer.window.add_dock_widget(wrapper, area="right", name="Macrophage Tools")
    main_tools_dock = curr_dock

    # Place the tools dock directly below the "Edit CD206+DAPI+Masks" dock
    try:
        qt_window = viewer.window._qt_window
        edit_dock = next(
            (d for d in qt_window.findChildren(QtWidgets.QDockWidget)
             if "Edit CD206" in (d.windowTitle() or "") and d is not curr_dock),
            None
        )
        if edit_dock is not None:
            qt_window.splitDockWidget(edit_dock, curr_dock, QtCore.Qt.Vertical)
    except Exception:
        pass

    _register_keyboard_shortcuts(viewer)

    def _on_tools_destroyed():
        global main_tools_dock
        main_tools_dock = None
    try:
        wrapper.destroyed.connect(_on_tools_destroyed)
    except Exception:
        pass

def _make_add_image_layer_widget():
    w = magicgui(
        add_image_layer,
        call_button="Load Image",
        image_path={"label": "Select image (.tif/.tiff)", "filter": "*.tif *.tiff"},
        channel_names={"label": "Channel names (comma-separated)", "value": "Collagen, F480, CD206, DAPI, Brightfield"},
    )
    for le in w.image_path.native.findChildren(QtWidgets.QLineEdit):
        le.setReadOnly(True)
        le.setText("")
        le.setPlaceholderText("No file selected")
    w.native.setStyleSheet(_widget_stylesheet())
    return w

def _make_add_mask_layer_widget():
    w = magicgui(
        add_mask_layer,
        call_button="Load Mask",
        mask_path={"label": "Select mask (.tif/.tiff)", "filter": "*.tif *.tiff"},
    )
    for le in w.mask_path.native.findChildren(QtWidgets.QLineEdit):
        le.setReadOnly(True)
        le.setText("")
        le.setPlaceholderText("No file selected")
    w.native.setStyleSheet(_widget_stylesheet())
    return w

def make_add_layer_from_tif_widget():
    img_w = _make_add_image_layer_widget() 
    mask_w = _make_add_mask_layer_widget()

    load_container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()
    load_container.setLayout(layout)

    layout.addWidget(img_w.native)
    layout.addWidget(mask_w.native)
    layout.addStretch(1)
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)

    load_container.setStyleSheet(_widget_stylesheet())
    return load_container

def make_add_layer_from_zarr_widget():
    w = magicgui(
        add_layer_from_zarr, 
        folder={"label": "Zarr folder", "mode": "d"}, 
        call_button="Load Zarr"
    )
    w.native.setStyleSheet(_widget_stylesheet())
    return w

def make_edit_overlay_all_widget():
    w = magicgui(edit_overlay_all, call_button="Edit CD206 + DAPI + Masks")
    w.native.setStyleSheet(_widget_stylesheet())
    return w

def make_run_watershed_for_all_rois_widget():
    w = magicgui(run_watershed_for_all_rois, call_button="Run Watershed for All ROIs")
    w.native.setStyleSheet(_widget_stylesheet())
    _set_call_button_tooltip(w, "Run 3D Watershed for all ROIs detected. Instead of previewing the result, the segmentation results will be directly written into the Masks layer with new object IDs.")
    return w

###### main function to edit overlay ######
def edit_overlay_all():
    """Open the Macrophage Tools dock. Does not load any data."""
    viewer = napari.current_viewer()

    if "Masks" in viewer.layers:
        if select_object not in getattr(viewer.layers["Masks"], "mouse_drag_callbacks", []):
            viewer.layers["Masks"].mouse_drag_callbacks.append(select_object)
        viewer.layers["Masks"].selected_object_id = None
        viewer.layers["Masks"].click_coords = None
        viewer.layers.selection.clear()
        viewer.layers.selection.add(viewer.layers["Masks"])

    _built_widgets()
