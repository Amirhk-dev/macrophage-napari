import numpy as np
import napari
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import label, binary_fill_holes, gaussian_filter
from skimage.filters import threshold_otsu

from .error import _layers_not_in_viewer_error
from .state import dataState
from .analysis import cells_analysis
import torch
import torch.nn.functional as F

###### selection ######
def select_object(layer, event): # callback function
    """A callable function that will be called (select the clicked object) when user clicks on layer"""
    if layer.data is not None:
        coords = np.round(event.position).astype(int)
        z, y, x = coords
        Z, H, W = layer.data.shape
        if not (0 <= z < Z and 0 <= y < H and 0 <= x < W):
            msg = f"Clicked outside image area"
            show_warning(msg)
            print(msg)
            return
        object_id = layer.data[z, y, x]
        try: 
            layer.selected_object_id = object_id
        except Exception as e:
            print("Error setting selected_object_id:", e)
            layer.selected_object_id = None
        if object_id != 0:
            layer.click_coords = (z, y, x)
            msg = f"Selected object ID {object_id} at ({layer.click_coords[0]}, {layer.click_coords[1]}, {layer.click_coords[2]})"
            show_info(msg)
            print(msg)
        else:
            msg = f"Clicked on background"
            show_info(msg)
            print(msg)
            viewer = napari.current_viewer()
            if "Selection" in viewer.layers:
                viewer.layers.remove(viewer.layers["Selection"])
    else:
        msg = f"No mask data"
        show_warning(msg)
        print(msg)

 
###### deletion ######
def delete_object(*args, **kwargs):
    """Delete the connected component of the object on the current slice"""
    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    
    layer = viewer.layers.selection.active
    if layer.name != "Masks":
        msg = f"Current active layer is {layer.name}, please select the Masks layer"
        show_warning(msg)
        print(msg)
        return
    else:   
        if layer.selected_object_id is None and layer.click_coords is None:
            msg = f"Please select an object to delete"
            show_info(msg)
            print(msg)
            return
        else:
            object_id = layer.selected_object_id
            if object_id == 0:
                msg = f"Please select an object to delete"
                show_info(msg)
                print(msg)
                return
            else:
                curr_position = layer.click_coords # coordinates of the click (z, y, x)
                curr_slice_idx = curr_position[0]
                layer_slice = layer.data[curr_slice_idx]
                # layer_slice[layer_slice == object_id] = 0 # only affect the current slice, delete the whole object
                mask_slice = (layer_slice == object_id) 
                labelled_mask_slice, num_features = label(mask_slice) # binary mask -> each connected component will be labelled differently
                target_label = labelled_mask_slice[curr_position[1], curr_position[2]]
                layer_slice[labelled_mask_slice == target_label] = 0 # delete only the connected component
                for l in viewer.layers:
                    l.refresh()
                msg = f"Deleted object {object_id} at slice {curr_slice_idx}"
                show_info(msg)
                print(msg)

    layer.selected_object_id = 0 
    layer.click_coords = None


def delete_all(*args, **kwargs):
    """Delete the object completely throughout the image"""
    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    
    layer = viewer.layers.selection.active
    if layer.name != "Masks":
        msg = f"Current active layer is {layer.name}, please select the Masks layer"
        show_warning(msg)
        print(msg)
        return  
    else: 
        object_id = layer.selected_object_id
        if object_id == 0:
            msg = f"Please select an object to delete"
            show_info(msg)
            print(msg)
            return
        else:
            layer.data[layer.data == object_id] = 0
            for l in viewer.layers:
                l.refresh()
            msg = f"Deleted object {object_id} at all slices"
    
    layer.selected_object_id = 0
    layer.click_coords = None


###### edit object id ######
def edit_object_id(new_id: int=None):
    """Change the object id of the connected component on the slice that the user clicked on or give it the next available id"""
    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
  
    layer = viewer.layers.selection.active  
    if layer.name != "Masks":
        msg = f"Current active layer is {layer.name}, please select the Masks layer"
        show_warning(msg)
        print(msg)
        return
    
    if layer.selected_object_id is None and layer.click_coords is None:
        msg = f"Please select an object to edit"
        show_info(msg)
        print(msg)
        return
    else:
        object_id = layer.selected_object_id
        if object_id == 0:
            msg = f"Please select an object to edit"
            show_info(msg)
            print(msg)
            return
        else:
            curr_position = layer.click_coords 
            curr_slice_idx = curr_position[0]
            layer_slice = layer.data[curr_slice_idx]
            mask_slice = (layer_slice == object_id) 
            labelled_mask_slice, num_features = label(mask_slice)
            target_label = labelled_mask_slice[curr_position[1], curr_position[2]]

            if new_id is not None:
                layer_slice[labelled_mask_slice == target_label] = new_id 
            else:
                max_id = layer.data.max()
                new_id = max_id + 1
                layer_slice[labelled_mask_slice == target_label] = new_id

            for l in viewer.layers:
                l.refresh()
            msg = f"Changed object {object_id} at slice {curr_slice_idx} to new ID {new_id}"
            show_info(msg)
            print(msg)
    
    layer.selected_object_id = 0
    layer.click_coords = None


def renumber():
    """Reassign the object ids in the image so that they are consecutive and ordered by their first appearance"""
    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    
    layer = viewer.layers.selection.active
    if layer.name == "Masks":
        old_to_new_mapping = {}
        next_id = 1
        for z in range(layer.data.shape[0]):
            slice = layer.data[z]
            slice_renumbered = np.zeros_like(np.array(slice))
            input_ids = np.unique(np.array(slice))
            input_ids = input_ids[input_ids != 0]
            for obj_id in input_ids:
                if obj_id not in old_to_new_mapping:
                    old_to_new_mapping[obj_id] = next_id
                    next_id += 1
                slice_renumbered[np.array(slice) == obj_id] = old_to_new_mapping[obj_id]
            layer.data[z] = slice_renumbered
            layer.refresh()
        msg = f"Renumbered labels from 1 to {int(layer.data.max())} sequentially"
        show_info(msg)
        print(msg)
    
    else:
        msg = f"Renumbering only works in the Masks layer"
        show_info(msg)
        print(msg)

    layer.selected_object_id = 0
    layer.click_coords = None


###### edit a specific object ######
def sync_object_to_masks():
    """To sync changes by napari's built-in tool (e.g., paint brush, label eraser) on the Object layer back to the Masks layer.
    Please press the Apply Changes button if you make any changes on the Object layer so that they will also be visible on the Masks layer.""" 

    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    
    object_layer = viewer.layers.selection.active
    if not object_layer.name.startswith("Object"):
        msg = f"Current active layer is not an Object layer, please select the object layer which contains the changes you want to save as active"
        show_warning(msg)
        print(msg)
    else:
        object_id = int(object_layer.name.split()[-1])
        mask = (object_layer.data > 0)
        mask_data = viewer.layers["Masks"].data.copy()

        mask_data[mask_data==object_id] = 0
        mask_data[mask] = object_id
        viewer.layers["Masks"].data = mask_data
        msg = f"Changes of object {object_id} applied. Delete any Object layers that are no longer needed"
        viewer.layers.selection.clear()
        viewer.layers.selection.add(viewer.layers["Masks"])
        object_layer_name = "Object " + str(object_id)
        viewer.layers[object_layer_name].visible = False
        viewer.layers["Masks"].visible = True
        show_info(msg)
        print(msg)


def add_object_layer(object_id: int = None):
    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    layer = viewer.layers["Masks"]
    viewer.layers.selection.clear()
    viewer.layers.selection.add(layer)

    target_object_id = None

    if layer.selected_object_id is not None and layer.click_coords is not None:
        if layer.selected_object_id != 0:
            target_object_id = layer.selected_object_id

    if target_object_id is None and object_id is not None:
        target_object_id = object_id

    if target_object_id is None:
        msg = f"Please specify an object to view"
        show_info(msg)
        print(msg)
    
    mask_data = layer.data # changes (using napari's built-in tools) on Masks layer will be synced to Object layer
    all_object_ids = np.unique(mask_data)
    max_object_id = mask_data.max()
    if target_object_id > max_object_id:
        msg = f"Only {max_object_id} macrophages in this image"
        show_warning(msg)
        print(msg) 
    elif target_object_id not in all_object_ids:
        msg = f"Macrophage with {target_object_id} does not exist"
        show_warning(msg)
        print(msg)
    else:   
        for l in viewer.layers:
            l.visible = False
        if "CD206" in viewer.layers:
            viewer.layers["CD206"].visible = True
        else: 
            viewer.add_image(dataState.cd206_images, name="CD206") 

        mask = (mask_data == target_object_id)
        object_position = np.where(mask)[0].min()
        layer_name = f"Object {target_object_id}"
        if layer_name not in viewer.layers:
            object_layer = viewer.add_labels(mask, name=layer_name)
        else:    
            viewer.layers[layer_name].data = mask
            viewer.layers[layer_name].visible = True

        msg = f"Object {target_object_id} first appears at slice {object_position}."
        show_info(msg)
        print(msg)
    
    layer.selected_object_id = 0
    layer.click_coords = None


###### walk through objects ######
def _sort_objects_by_xy(mask_layer, z: int) -> list[int]:
    """Return object ids on slice z, sorted by top-left (ymin, xmin)."""
    slice = np.asarray(mask_layer.data[z])
    ids = np.unique(slice)
    ids = ids[ids != 0]
    items = []
    for obj_id in ids:
        ys, xs = np.where(slice == obj_id)
        if ys.size == 0:
            continue
        y0, x0 = int(ys.min()), int(xs.min())
        items.append((y0, x0, int(obj_id)))
    items.sort(key=lambda t: (t[0], t[1]))
    return [oid for _, _, oid in items]


def _add_highlight_layer(viewer):
    if "Selection" not in viewer.layers:
        return viewer.add_shapes(
            name="Selection",
            shape_type="rectangle",
            edge_color="yellow",
            face_color="transparent",
            edge_width=3,
            opacity=1.0,
            blending="translucent",
        )
    return viewer.layers["Selection"]


def _highlight_object_on_slice(mask_layer, obj_id: int, z: int):
    """Draw a yellow rectangle around the object to highlight the object."""
    viewer = napari.current_viewer()
    sel_layer = _add_highlight_layer(viewer)
    if sel_layer in viewer.layers.selection:
        viewer.layers.selection.remove(sel_layer)
    viewer.layers.selection.add(mask_layer)
    try:
        sel_layer.data = []
    except Exception:
        pass

    data = mask_layer.data
    if not (0 <= z < data.shape[0]):
        return
    ys, xs = np.where(data[z] == obj_id)
    if ys.size == 0:
        return
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    curr_bbox = np.array([
        [z, y0, x0],
        [z, y0, x1],
        [z, y1, x1],
        [z, y1, x0],
    ], dtype=float)
    sel_layer.add(
        curr_bbox,
        shape_type="rectangle",
        edge_color="yellow",
        face_color="transparent",
        edge_width=3,
    )
    if z < data.shape[0] - 1: # simulate a change of slice to force refresh
        viewer.dims.set_current_step(0, z+1)
    else:
        viewer.dims.set_current_step(0, z-1)
    viewer.dims.set_current_step(0, z)


def _activate_object_in_slice(viewer, mask_layer, obj_id: int, z: int):
    """Save the coords and highlight the object"""
    slice = np.asarray(mask_layer.data[z])
    ys, xs = np.where(slice == obj_id)
    if ys.size == 0:
        return
    y = int(ys.min()); x = int(xs.min()) 
    mask_layer.selected_object_id = obj_id
    mask_layer.click_coords = (z, y, x)
    try:
        viewer.dims.set_current_step(0, z)
    except Exception:
        pass
    _highlight_object_on_slice(mask_layer, obj_id, z)
    show_info(f"Object {obj_id} at z={z}")


def _step_object_in_slice(delta: int):
    """delta=+1 next; -1 previous"""
    viewer = napari.current_viewer()
    required_layers = ["Masks"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    curr_layer = viewer.layers.selection.active
    if curr_layer.name != "Masks":
        show_warning("Only works in Masks layer")
        return
    mask_layer = viewer.layers["Masks"]
    z = int(viewer.dims.current_step[0])

    ids = _sort_objects_by_xy(mask_layer, z)
    if not ids:
        show_info("No objects on this slice")
        return

    curr_obj_id = getattr(mask_layer, "selected_object_id", None)
    if curr_obj_id in ids:
        curr_obj_idx = ids.index(int(curr_obj_id))
        next_obj_idx = (curr_obj_idx + delta) % len(ids)
    else:
        if delta > 0:
            next_obj_idx = 0
        else:
            next_obj_idx = len(ids) - 1
    _activate_object_in_slice(viewer, mask_layer, ids[next_obj_idx], z)


##### Interpolate image #####
def interpolate_to_isotropic():
    if not dataState.voxel_size_um:
        show_warning("Voxel size is not set yet. Please set voxel size first.")
        return
    old_voxel_size = dataState.voxel_size_um
    old_shape = dataState.cd206_images.shape
    physical_size = [old_shape[i]*old_voxel_size[i] for i in range(dataState.cd206_images.ndim)]

    new_voxel_size = min(old_voxel_size)
    new_shape = [int(round(physical_size[i]/new_voxel_size)) for i in range(dataState.cd206_images.ndim)]

    viewer = napari.current_viewer()
    curr_layer = viewer.layers.selection.active
    n_active_layers = len(viewer.layers.selection)
    if n_active_layers != 1:
        show_warning("Please select only one layer (CD206/Mask/DAPI) to interpolate")
        return
    if curr_layer.name == "CD206" or curr_layer.name == "DAPI":
        if curr_layer.name == "CD206":
            image = dataState.cd206_images
        elif curr_layer.name == "DAPI":
            image = dataState.dapi_images
        image_torch = torch.from_numpy(image).to(torch.float64)  # convert ndarray to tensor, cast to float64
        isotropic_img = F.interpolate(
            image_torch.unsqueeze(0).unsqueeze(0), # add batch and channel dims [1,1,Z,H,W]
            size = new_shape, # output spatial size
            mode = "trilinear"
        )
        isotropic_img = isotropic_img.squeeze(0).squeeze(0).numpy()
        viewer.add_image(isotropic_img, name=f"{curr_layer.name} (iso)", blending="additive", colormap=curr_layer.colormap.name)
    elif curr_layer.name == "Masks":
        mask = viewer.layers["Masks"].data
        mask_torch = torch.from_numpy(mask).to(torch.float64)
        isotropic_img = F.interpolate(
            mask_torch.unsqueeze(0).unsqueeze(0),
            size = new_shape,
            mode = "nearest"
        )
        isotropic_img = isotropic_img.squeeze(0).squeeze(0).to(torch.int32).numpy()
        viewer.add_labels(isotropic_img, name="Masks (iso)", blending="additive")
    else:
        show_warning("Please select either CD206, DAPI or Masks layer to interpolate")
        return


##### Shrink mask to CD206 + DAPI boundaries (current slice only) #####
def shrink_mask_to_cd206():
    """Shrink the selected object's mask on the current slice using CD206 + DAPI signal.

    Region-based approach:
    1. Build a combined signal from CD206 (+ DAPI if available), normalised to [0, 1].
    2. Compute an Otsu threshold on the signal values *inside the mask only*, so the
       threshold separates bright cell signal from dim regions included in the mask.
    3. Keep only pixels above the threshold, then retain the largest connected component
       (the real cell body) and discard small fragments.
    The result is always intersected with the original mask so it can only shrink.
    """
    viewer = napari.current_viewer()
    required_layers = ["Masks", "CD206"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return

    layer = viewer.layers.selection.active
    if layer.name != "Masks":
        msg = f"Current active layer is {layer.name}, please select the Masks layer"
        show_warning(msg)
        print(msg)
        return

    object_id = getattr(layer, "selected_object_id", None)
    if not object_id:
        msg = "Please click on an object to select it first"
        show_info(msg)
        print(msg)
        return

    cd206 = dataState.cd206_images
    if cd206 is None:
        show_warning("CD206 image not loaded")
        return

    mask_data = layer.data
    z = int(viewer.dims.current_step[0])

    obj_mask_slice = (mask_data[z] == object_id)
    if not obj_mask_slice.any():
        show_warning(f"Object {object_id} is not present on slice {z}")
        return

    def _norm(arr):
        a = arr.astype(float)
        lo, hi = a.min(), a.max()
        if hi == lo:
            return None
        return (a - lo) / (hi - lo)

    cd206_norm = _norm(cd206[z])
    if cd206_norm is None:
        show_warning(f"No CD206 intensity variation on slice {z}")
        return

    def _otsu_mask(channel_norm):
        """Return a boolean mask of pixels above Otsu threshold, computed inside the object mask."""
        vals = channel_norm[obj_mask_slice]
        if vals.max() == vals.min():
            return None
        thresh = threshold_otsu(vals)
        return channel_norm >= thresh

    cd206_above = _otsu_mask(cd206_norm)

    # Threshold DAPI independently and union with CD206 so neither signal is lost
    dapi = dataState.dapi_images
    channels_used = "CD206"
    if dapi is not None:
        dapi_norm = _norm(dapi[z])
        if dapi_norm is not None:
            dapi_above = _otsu_mask(dapi_norm)
            if dapi_above is not None:
                above_thresh = (cd206_above | dapi_above) & obj_mask_slice
                channels_used = "CD206 + DAPI"
            else:
                above_thresh = cd206_above & obj_mask_slice
        else:
            above_thresh = cd206_above & obj_mask_slice
    else:
        above_thresh = cd206_above & obj_mask_slice

    if not above_thresh.any():
        show_warning(f"No pixels above threshold for object {object_id} on slice {z}. Aborting.")
        return

    # Keep the largest connected component so we get the main cell body
    labeled, n_comps = label(above_thresh)
    if n_comps > 1:
        comp_sizes = np.bincount(labeled.ravel())
        comp_sizes[0] = 0  # ignore background label
        above_thresh = (labeled == int(comp_sizes.argmax()))

    # Fill interior holes (background pixels fully enclosed by the object)
    above_thresh = binary_fill_holes(above_thresh)

    # Smooth the boundary: blur the binary mask and re-threshold at 0.5
    above_thresh = gaussian_filter(above_thresh.astype(float), sigma=2) >= 0.5

    shrunk = above_thresh & obj_mask_slice  # safety: intersect with original mask

    if not shrunk.any():
        show_warning(f"Shrinking would remove all voxels of object {object_id} on slice {z}. Aborting.")
        return

    # Write back: clear old object pixels, paint shrunk region
    new_slice = mask_data[z].copy()
    new_slice[obj_mask_slice] = 0
    new_slice[shrunk] = object_id
    mask_data[z] = new_slice

    for l in viewer.layers:
        l.refresh()

    removed_count = int(obj_mask_slice.sum()) - int(shrunk.sum())
    msg = f"Shrunk object {object_id} on slice {z} using {channels_used}: removed {removed_count} voxels"
    show_info(msg)
    print(msg)

    layer.selected_object_id = 0
    layer.click_coords = None