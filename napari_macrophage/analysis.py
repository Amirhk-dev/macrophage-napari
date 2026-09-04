import napari
import numpy as np

from napari.utils.notifications import show_info, show_warning
from qtpy import QtWidgets
from skimage import measure

from .error import _layers_not_in_viewer_error
from .state import dataState, get_voxel_size_um
from .ui import _widget_stylesheet


###### show cell analysis ######
# voxel arguments are in ZYX order to match state.voxel_size_um = (dz, dy, dx).
def _compute_volume(dz: float, dy: float, dx: float, mask_3d: np.ndarray):
    voxel_volume = float(dz) * float(dy) * float(dx)

    labels = np.unique(mask_3d)
    labels = labels[labels != 0]
    volume_stats: dict[int, float] = {}

    for label in labels:
        voxel_count = int(np.count_nonzero(mask_3d == label))
        volume = voxel_count * voxel_volume
        volume_stats[int(label)] = float(volume)
    return volume_stats


def _compute_sphericity(dz: float, dy: float, dx: float, mask_3d: np.ndarray):
    """ Ψ = (pi^1/3 * (6V)^2/3) / A"""
    voxel_volume = float(dz) * float(dy) * float(dx)

    labels = np.unique(mask_3d)
    labels = labels[labels != 0]
    sphericity_stats: dict[int, float] = {}
    surface_area_stats: dict[int, float] = {}

    for label in labels:
        zz, yy, xx = np.where(mask_3d == label)
        if zz.size == 0:
            sphericity_stats[int(label)] = 0.0
            continue
        z0, z1 = int(zz.min()), int(zz.max())
        y0, y1 = int(yy.min()), int(yy.max())
        x0, x1 = int(xx.min()), int(xx.max())
        sub = (mask_3d[z0:z1+1, y0:y1+1, x0:x1+1] == label).astype(np.uint8)

        V = float(np.count_nonzero(sub)) * voxel_volume
        if V <= 0:
            sphericity_stats[int(label)] = 0.0
            continue
        try:
            # sub axes are (Z, Y, X) so spacing must be (dz, dy, dx).
            verts, faces, _, _ = measure.marching_cubes(sub, spacing=(dz, dy, dx))
            A = float(measure.mesh_surface_area(verts, faces)) # compute surface area, given vertices and triangular faces
            if A <= 0:
                sphericity = 0.0
                A = 0.0
            else:
                sphericity = float((np.pi**(1.0/3.0)) * ((6.0*V)**(2.0/3.0)) / A)
                if sphericity > 1.0:
                    sphericity = 1.0
        except Exception:
            sphericity = 0.0
            A = 0.0
        sphericity_stats[int(label)] = sphericity
        surface_area_stats[int(label)] = A
    return sphericity_stats, surface_area_stats


def cells_analysis(*args, **kwargs):
    viewer = napari.current_viewer()
    required_layers = ["Masks"]
    if _layers_not_in_viewer_error(viewer, required_layers):
        return
    mask_data = np.asarray(viewer.layers["Masks"].data)

    voxel_size = get_voxel_size_um()
    if voxel_size is None or any(float(v) <= 0.0 for v in voxel_size):
        show_warning("Pixel size is not set. Please set Pixel size X, Y, Z first.")
        return
    vol = _compute_volume(*voxel_size, mask_data) # unpack tuple as arguments
    sph, surface_area = _compute_sphericity(*voxel_size, mask_data)

    labels = sorted(set(vol.keys()) | set(sph.keys()))
    if not labels:
        show_info("No cells found in Masks")
        return

    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Cells Analysis")
    layout = QtWidgets.QVBoxLayout(dlg)

    table = QtWidgets.QTableWidget(len(labels), 5, dlg) # the entire table widget
    table.setHorizontalHeaderLabels(["Label ID", "Volume [µm3]", "Surface Area [µm2]", "Sphericity [-]", "Note"])
    table.verticalHeader().setVisible(False)

    for row, lid in enumerate(labels):
        v = vol.get(lid, 0.0)
        s = sph.get(lid, 0.0)
        a = surface_area.get(lid, 0.0)
        # print(row, lid, v, s)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(lid))) # an object representing a single cell in the table
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{v:.3f}"))
        table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{a:.3f}"))
        table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{s:.3f}"))
        note = ""
        if a < 200.0:
            note = "* small object, surface area approximation not so accurate"
        table.setItem(row, 4, QtWidgets.QTableWidgetItem(note))
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)
    layout.addWidget(table)

    btns = QtWidgets.QHBoxLayout()
    btn_save = QtWidgets.QPushButton("Save CSV")
    btn_close = QtWidgets.QPushButton("Close")
    btns.addWidget(btn_save)
    btns.addStretch(1)
    btns.addWidget(btn_close)
    layout.addLayout(btns)

    def _save_csv():
        default_name = (dataState.file_name or "cells") + "_cell_analysis"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            dlg, "Save Cells Analysis", default_name, "CSV Files (*.csv);;All Files (*)"
        ) # QtWidgets.QFileDialog.getSaveFileName(parent, caption, directory, filter)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                # Column order must match the row layout below (and the on-screen table).
                f.write("Label ID, Volume [\u00B5m3], Surface Area [\u00B5m2], Sphericity [-], Note\n")
                for lid in labels:
                    v = vol.get(lid, 0.0)
                    a = surface_area.get(lid, 0.0)
                    s = sph.get(lid, 0.0)
                    note = ""
                    if a < 200.0:
                        note = "* small object, surface area approximation not so accurate"
                    f.write(f"{lid},{v:.3f},{a:.3f},{s:.3f},{note}\n")

            show_info(f"Saved CSV to {path}")
        except Exception as e:
            show_warning(f"Failed to save CSV: {e}")

    btn_save.clicked.connect(_save_csv) # call _save_csv when click btn_save
    btn_close.clicked.connect(dlg.close)
    try:
        dlg.setStyleSheet(_widget_stylesheet())
    except Exception:
        pass
    dlg.resize(520, 400)
    dlg.show()
    dlg.exec_()