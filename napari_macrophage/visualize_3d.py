import struct
from pathlib import Path

import napari
import numpy as np
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy import QtWidgets
from qtpy.QtWidgets import QApplication

from .error import _layers_not_in_viewer_error
from .macrophage_mesh import mesh_from_binary
from .state import get_voxel_size_um


LAYER_PREFIX = "3D Object"


def _layer_name(object_id: int) -> str:
    return f"{LAYER_PREFIX} {object_id}"


def _write_obj(path: Path, verts_xyz: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# napari-macrophage export\n")
        for v in verts_xyz:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces + 1:  # OBJ is 1-indexed
            f.write(f"f {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _write_ply(path: Path, verts_xyz: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts_xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts_xyz:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _write_stl(path: Path, verts_xyz: np.ndarray, faces: np.ndarray) -> None:
    tris = verts_xyz[faces]  # (M, 3, 3)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = (normals / norms).astype(np.float32)

    with open(path, "wb") as f:
        f.write(b"\x00" * 80)  # header
        f.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            f.write(normals[i].astype("<f4").tobytes())
            f.write(tris[i].astype("<f4").tobytes())
            f.write(b"\x00\x00")  # attribute byte count


def _set_canvas_bg(viewer, color: str) -> None:
    """Set viewer canvas background color (best-effort across napari versions)."""
    try:
        viewer.window._qt_viewer.canvas.bgcolor = color
    except Exception as e:
        show_warning(f"Could not change background color: {e}")


def _screenshot_viewer(viewer) -> None:
    """Save a PNG of the viewer canvas (canvas only, no UI chrome)."""
    path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,
        "Save 3D Screenshot",
        "3d_object.png",
        "PNG (*.png)",
    )
    if not path_str:
        return
    path = Path(path_str)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    try:
        viewer.screenshot(str(path), canvas_only=True, flash=False)
        show_info(f"Screenshot saved to {path}")
    except Exception as e:
        show_error(f"Failed to save screenshot: {e}")


def _export_surface_from_viewer(viewer):
    """Save the (first) Surface layer in ``viewer`` as STL/OBJ/PLY."""
    surfaces = [l for l in viewer.layers if isinstance(l, napari.layers.Surface)]
    if not surfaces:
        show_error("No 3D surface in this window to export.")
        return
    active = viewer.layers.selection.active
    layer = active if isinstance(active, napari.layers.Surface) else surfaces[0]

    verts_zyx, faces = layer.data[0], layer.data[1]
    # napari verts are (z, y, x) in µm; standard mesh formats want (x, y, z).
    verts_xyz = np.asarray(verts_zyx)[:, [2, 1, 0]].astype(np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,
        "Export 3D Mesh",
        layer.name.replace(" ", "_"),
        "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)",
    )
    if not path_str:
        return

    path = Path(path_str)
    ext = path.suffix.lower()
    if ext not in (".stl", ".obj", ".ply"):
        path = path.with_suffix(".stl")
        ext = ".stl"

    try:
        if ext == ".obj":
            _write_obj(path, verts_xyz, faces)
        elif ext == ".ply":
            _write_ply(path, verts_xyz, faces)
        else:
            _write_stl(path, verts_xyz, faces)
        show_info(f"Exported mesh to {path}")
    except Exception as e:
        show_error(f"Failed to export mesh: {e}")


def visualize_macrophage_3d(
    object_id: int = 1,
    smooth_iter: int = 50,
    pre_smooth_sigma: float = 1.0,
    taper_ends: bool = True,
    shading: str = "smooth",
    crop_to_object: bool = True,
    background: str = "black",
):
    """Generate a publication-quality 3D surface mesh for a single macrophage.

    Opens the mesh in a **new napari window** dedicated to that object, so it
    is not affected by the main viewer's coordinate frame or other layers.

    Args:
        object_id: Label ID of the macrophage to visualise (must exist in Masks).
        smooth_iter: Laplacian smoothing iterations (higher = smoother surface).
        pre_smooth_sigma: Gaussian sigma (voxels) applied before marching cubes.
        taper_ends: Round the top/bottom Z slices instead of a flat-cut cap.
        shading: napari surface shading — ``"smooth"``, ``"flat"``, or ``"none"``.
        crop_to_object: Centre the mesh at the origin. Physical µm scale is
            preserved, so different objects remain directly comparable in size.
        background: Canvas background colour — ``"black"`` or ``"white"``.
            Can also be toggled from a button in the new window.
    """
    main_viewer = napari.current_viewer()
    if _layers_not_in_viewer_error(main_viewer, ["Masks"]):
        return

    voxel = get_voxel_size_um()
    if voxel is None:
        show_warning("Voxel size is not set. Please set voxel size first.")
        return

    mask = np.asarray(main_viewer.layers["Masks"].data)
    if object_id not in np.unique(mask):
        show_error(f"Object ID {object_id} not found in Masks layer.")
        return

    show_info(f"Generating 3D mesh for object {object_id}… (this may take a few seconds)")
    QApplication.processEvents()

    binary = (mask == object_id)
    result = mesh_from_binary(
        binary,
        spacing=voxel,
        smooth_iter=int(smooth_iter),
        pre_smooth_sigma=float(pre_smooth_sigma),
        taper_ends=bool(taper_ends),
        cap_ends=True,
        fill_mesh_holes=True,
        resample_isotropic=True,
    )
    if result is None:
        show_error(f"Could not build mesh for object {object_id} (empty after preprocessing).")
        return

    verts, faces = result

    if crop_to_object:
        centre = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
        verts = (verts - centre).astype(np.float32)

    values = np.full(len(verts), float(object_id), dtype=np.float32)

    # Open a fresh napari window for this object — clean, isolated 3D view.
    viewer_3d = napari.Viewer(ndisplay=3, title=f"3D Object {object_id}")
    viewer_3d.add_surface(
        (verts, faces, values),
        name=_layer_name(object_id),
        shading=shading,
        colormap="magma",
    )
    viewer_3d.reset_view()
    _set_canvas_bg(viewer_3d, background)

    bg_state = {"white": background == "white"}
    def _toggle_bg():
        bg_state["white"] = not bg_state["white"]
        _set_canvas_bg(viewer_3d, "white" if bg_state["white"] else "black")

    controls = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout(controls)
    vbox.setContentsMargins(4, 4, 4, 4)
    vbox.setSpacing(4)
    screenshot_btn = QtWidgets.QPushButton("Save PNG (canvas only)")
    screenshot_btn.clicked.connect(lambda: _screenshot_viewer(viewer_3d))
    export_btn = QtWidgets.QPushButton("Export Mesh (STL / OBJ / PLY)")
    export_btn.clicked.connect(lambda: _export_surface_from_viewer(viewer_3d))
    bg_btn = QtWidgets.QPushButton("Toggle Background (black/white)")
    bg_btn.clicked.connect(_toggle_bg)
    vbox.addWidget(screenshot_btn)
    vbox.addWidget(export_btn)
    vbox.addWidget(bg_btn)
    vbox.addStretch(1)
    viewer_3d.window.add_dock_widget(controls, area="right", name="Publication Tools")

    extent_um = verts.max(axis=0) - verts.min(axis=0)
    show_info(
        f"3D window opened for object {object_id}: "
        f"{len(verts)} vertices, {len(faces)} faces, "
        f"extent {extent_um[2]:.1f}×{extent_um[1]:.1f}×{extent_um[0]:.1f} µm (X×Y×Z)."
    )


