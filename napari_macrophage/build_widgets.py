"""Build the docked "Macrophage Tools" widget and its menu-entry factories.

Each ``make_*_widget`` function is referenced from ``napari.yaml`` and returns
a Qt widget that napari mounts as a dock. :func:`_built_widgets` assembles the
main scrollable panel with all editing, segmentation, bbox, and analysis
controls plus the in-plugin log at the bottom.
"""

import napari
from magicgui import magicgui
from qtpy import QtCore, QtWidgets

from .analysis import cells_analysis
from .bbox import (
    add_roi_layer,
    detect_objects_with_onnx,
    export_bboxes_to_yolo,
    generate_bboxes_from_mask_layer,
    import_bboxes_from_yolo_folder,
)
from .clean3d import clean_3d_macrophages
from .edit_mask_image import (
    _step_object_in_slice,
    add_object_layer,
    delete_all,
    delete_object,
    edit_object_id,
    interpolate_to_isotropic,
    renumber,
    select_object,
    shrink_mask_to_cd206,
    sync_object_to_masks,
)
from .io import add_image_layer, add_layer_from_zarr, add_mask_layer
from .log_panel import LogPanel
from .segmentation import (
    _create_slider_or_update_otsu,
    finalise_mask,
    run_watershed_for_all_rois,
    run_watershed_on_bbox,
)
from .state import set_voxel_size_um
from .ui import CollapsibleSection, _disable_wheel_on_inputs, _set_call_button_tooltip, _widget_stylesheet
from .visualize_3d import visualize_macrophage_3d


def _register_keyboard_shortcuts(viewer):
    """Bind the plugin's mask-editing hotkeys (d, Shift+D, v, i, Up/Down) on ``viewer``."""
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
        nms_iou_threshold={"label": "NMS IoU threshold", "min": 0.0, "max": 1.0, "step": 0.05},
        current_slice_only={"label": "Current slice only"},
    )
    for le in detect_onnx_widget.onnx_path.native.findChildren(QtWidgets.QLineEdit):
        le.setReadOnly(True)
        le.setText("")
        le.setPlaceholderText("Default model (select to override)")

    image_info_widget = magicgui(
        set_voxel_size_um,
        voxel_x={"label": "Pixel size X [µm]"},
        voxel_y={"label": "Pixel size Y [µm]"},
        voxel_z={"label": "Pixel size Z [µm]"},
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

    clean_3d_widget = magicgui(
        clean_3d_macrophages,
        erosion_radius={"label": "2D erosion radius", "min": 0, "max": 20, "step": 1},
        closing_radius_2d={"label": "2D closing radius", "min": 0, "max": 20, "step": 1},
        gaussian_sigma={"label": "Gaussian sigma", "min": 0.0, "max": 10.0, "step": 0.1},
        closing_radius_3d={"label": "3D closing radius", "min": 0, "max": 10, "step": 1},
        min_voxels_3d={"label": "Min voxels (3D)", "min": 0, "max": 1000000, "step": 50},
        call_button="Clean 3D Macrophages",
    )
    for spin in clean_3d_widget.native.findChildren(QtWidgets.QSpinBox):
        spin.setStyleSheet("font-size: 10pt;")
    for spin in clean_3d_widget.native.findChildren(QtWidgets.QDoubleSpinBox):
        spin.setStyleSheet("font-size: 10pt;")

    visualize_3d_widget = magicgui(
        visualize_macrophage_3d,
        object_id={"label": "Object ID", "min": 1, "step": 1},
        smooth_iter={"label": "Smoothing iterations", "min": 0, "max": 500, "step": 5},
        pre_smooth_sigma={"label": "Pre-smooth sigma", "min": 0.0, "max": 3.0, "step": 0.1},
        taper_ends={"label": "Round ends (taper)"},
        shading={"label": "Shading", "choices": ["smooth", "flat", "none"]},
        crop_to_object={"label": "Crop & center (µm)"},
        background={"label": "Background", "choices": ["black", "white"]},
        call_button="Generate 3D",
    )
    for spin in visualize_3d_widget.native.findChildren(QtWidgets.QSpinBox):
        spin.setStyleSheet("font-size: 10pt;")
    for spin in visualize_3d_widget.native.findChildren(QtWidgets.QDoubleSpinBox):
        spin.setStyleSheet("font-size: 10pt;")


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
    _set_call_button_tooltip(detect_onnx_widget, "Run ONNX object detection on all Z slices using CD206 + DAPI (CPU). Per-slice NMS drops the smaller of any two overlapping boxes above 'NMS IoU threshold'. Results are added to the ROI layer and can be edited or exported like any other bounding box.")

    _set_call_button_tooltip(image_info_widget, "Update the voxel size information used for analysis and processing.")
    _set_call_button_tooltip(cells_analysis_widget, "Compute volume and sphericity for each cell based on Masks layer.")
    _set_call_button_tooltip(interpolate_widget, "Interpolate the current image/mask layer to isotropic voxel size.")
    _set_call_button_tooltip(renumber_widget, "Renumber all objects to consecutive IDs by their first appearance in the image.")

    _set_call_button_tooltip(visualize_3d_widget, "Generate a smoothed 3D surface mesh for the selected object ID and open it in a NEW napari window. Each generation gets its own isolated 3D view. When 'Crop & center' is on, the mesh is centered at the origin (physical µm scale preserved — different objects remain comparable). An 'Export Mesh' button appears inside the new window for saving as STL/OBJ/PLY.")

    _set_call_button_tooltip(clean_3d_widget, "Clean every labeled macrophage in the Masks layer independently. Per slice: binary erosion → keep largest 2D component → fill holes → 2D closing → Gaussian smoothing. Then per object: optional 3D closing → keep largest 3D connected component (26-conn) → drop if smaller than 'Min voxels (3D)'. Original IDs are preserved.")

    def _row(*widgets: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
        """Return a tightly-spaced horizontal layout containing ``widgets``."""
        r = QtWidgets.QHBoxLayout()
        r.setSpacing(4)
        for w in widgets:
            layout = w.layout()
            if layout is not None:
                layout.setContentsMargins(4, 4, 4, 4)
                layout.setSpacing(4)
            r.addWidget(w)
        return r

    def _section(title: str, *rows: QtWidgets.QLayout | QtWidgets.QWidget) -> CollapsibleSection:
        """Build a collapsed :class:`CollapsibleSection` populated with ``rows``.

        Each entry in ``rows`` may be a layout (added with ``addLayout``) or a
        widget (added with ``addWidget`` after tightening its own layout).
        """
        sec = CollapsibleSection(title, expanded=False)
        v = QtWidgets.QVBoxLayout()
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        for item in rows:
            if isinstance(item, QtWidgets.QLayout):
                v.addLayout(item)
            else:
                sub = item.layout()
                if sub is not None:
                    sub.setContentsMargins(4, 4, 4, 4)
                    sub.setSpacing(4)
                v.addWidget(item)
        sec.setContentLayout(v)
        return sec

    object_group = _section(
        "Objects",
        _row(delete_this_widget.native, delete_all_widget.native),
        _row(change_id_widget.native, new_id_widget.native),
        _row(view_object_widget.native, apply_changes_widget.native),
        _row(shrink_mask_widget.native),
    )

    seg_group = _section(
        "Segmentation",
        _row(add_roi_widget.native, finalise_3d_widget.native),
        _row(watershed_widget.native, finalise_watershed_widget.native),
        _row(slider_native),
    )

    bbox_group = _section(
        "Bounding Boxes (BBs)",
        _row(bboxes_widget.native, gen_bboxes_widget.native),
        _row(export_yolo_widget.native, import_yolo_widget.native),
        _row(detect_onnx_widget.native),
    )

    voxel_group = _section("Voxel Size", image_info_widget.native)

    analysis_processing_group = _section(
        "Analysis and Processing",
        renumber_widget.native,
        cells_analysis_widget.native,
        interpolate_widget.native,
    )

    cleaning_group = _section("Cleaning", clean_3d_widget.native)

    visualize_3d_group = _section("3D Visualization", visualize_3d_widget.native)

    # scrollable root container
    content = QtWidgets.QWidget()
    root = QtWidgets.QVBoxLayout(content)
    root.setContentsMargins(4, 4, 4, 4)
    root.setSpacing(4)
    for g in (object_group, seg_group, bbox_group, voxel_group, analysis_processing_group, visualize_3d_group, cleaning_group):
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
    wl.addWidget(scroll, 3)

    log_panel = LogPanel()
    wl.addWidget(log_panel, 1)

    wrapper.setStyleSheet(local_style)
    _disable_wheel_on_inputs(wrapper)

    curr_dock = viewer.window.add_dock_widget(wrapper, area="right", name="Macrophage Tools")
    main_tools_dock = curr_dock

    # Place the tools dock directly below the "Annotate & Correct Masks/Boxes" dock
    try:
        qt_window = viewer.window._qt_window
        edit_dock = next(
            (d for d in qt_window.findChildren(QtWidgets.QDockWidget)
             if "Annotate" in (d.windowTitle() or "") and d is not curr_dock),
            None
        )
        if edit_dock is not None:
            qt_window.splitDockWidget(edit_dock, curr_dock, QtCore.Qt.Vertical)
    except Exception:
        pass

    _register_keyboard_shortcuts(viewer)

    def _on_tools_destroyed():
        """Clear the module-level cache of the tools dock when it is destroyed."""
        global main_tools_dock
        main_tools_dock = None
    try:
        wrapper.destroyed.connect(_on_tools_destroyed)
    except Exception:
        pass

def _make_add_image_layer_widget():
    """Build the "Load Image" magicgui form used by the Load menu entry."""
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
    """Build the "Load Mask" magicgui form used by the Load menu entry."""
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
    """Return a Qt container that combines the Load Image and Load Mask forms.

    Registered in ``napari.yaml`` as the "Load Image + Mask" menu command.
    """
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
    _disable_wheel_on_inputs(load_container)
    return load_container

def make_add_layer_from_zarr_widget():
    """Return the "Load Zarr" magicgui widget for the Load menu."""
    w = magicgui(
        add_layer_from_zarr,
        folder={"label": "Zarr folder", "mode": "d"},
        call_button="Load Zarr"
    )
    w.native.setStyleSheet(_widget_stylesheet())
    _disable_wheel_on_inputs(w)
    return w

def make_edit_overlay_all_widget():
    """Return the widget that opens the full Macrophage Tools dock.

    Registered in ``napari.yaml`` as the "Annotate & Correct Masks/Boxes"
    menu command.
    """
    w = magicgui(edit_overlay_all, call_button="Annotate & Correct Masks/Boxes")
    w.native.setStyleSheet(_widget_stylesheet())
    return w

def make_run_watershed_for_all_rois_widget():
    """Return the "Run Watershed for All ROIs" batch widget."""
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
