"""
macrophage_mesh — standalone voxel→surface-mesh code for 3-D macrophage labels.

Extracted from the ``macrophage-image-processor`` repo (``src/visualize/rendering.py``
and ``src/visualize/macrophage_renderer.py``) so it can be dropped into a napari
plugin without pulling in the ``src.*`` package or PyVista.

The mesh pipeline is unchanged from the original — same order, same defaults::

    end taper → cap ends → isotropic Z resample → gaussian anti-alias
    → marching cubes → hole fill (optional) → laplacian smoothing

What is *different* from the original, and why:

* Returns plain ``(verts, faces)`` numpy arrays instead of a ``pyvista.PolyData``,
  so napari can consume them directly via ``viewer.add_surface``. PyVista is
  optional (only used for ``fill_holes``); smoothing falls back to a pure-scipy
  Laplacian that matches VTK's ``vtkSmoothPolyDataFilter`` relaxation scheme.
* Vertices come back in **(z, y, x) order, in physical units** (µm), matching
  napari's axis convention. The original handed the same array to PyVista, which
  reinterprets it as (x, y, z) — harmless for a screenshot, wrong for napari.
* Each label is cropped to its bounding box before meshing (the original meshed
  the full-size volume per cell, which is slow). Vertices are offset back into
  whole-volume coordinates, so geometry is identical.
* The original applied ``mask[:, ::-1, :]`` (a Y flip) purely to orient the
  PyVista camera. That is *not* applied here — it would misplace the mesh
  relative to the image layer. Set ``flip_y=True`` if you need the old look.

Dependencies: numpy, scipy, scikit-image. Optional: pyvista, napari.

Quick start
-----------
    import tifffile, napari
    from macrophage_mesh import label_volume_to_meshes, add_meshes_to_napari

    labels  = tifffile.imread('macrophage_labels.tif')     # (Z, Y, X) int
    spacing = (pixel_size_z, pixel_size_xy, pixel_size_xy) # µm, (dz, dy, dx)

    meshes = label_volume_to_meshes(labels, spacing)       # {cell_id: (verts, faces)}

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_labels(labels, scale=spacing)
    add_meshes_to_napari(viewer, meshes)
    napari.run()
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, gaussian_filter, zoom
from skimage import measure

__all__ = [
    "MESH_DEFAULTS",
    "mesh_from_binary",
    "label_volume_to_meshes",
    "add_meshes_to_napari",
    "meshes_to_napari_surface",
    "to_pyvista",
    "touches_xy_boundary",
]


# Values from configs/one_image_processor.yaml (`rendering:` block) — these are
# what produced output/.../renders_3d. Keep them if you want matching meshes.
MESH_DEFAULTS = dict(
    smooth_iter=50,
    pre_smooth_sigma=1.0,
    taper_ends=True,
    taper_n_slices=4,
    taper_max_iters=8,
    taper_disk_radius=2,
    taper_max_fraction=0.4,
)


# ── volume preprocessing ──────────────────────────────────────────────────────

def _apply_end_taper(vol, n_slices=4, max_iters=5, disk_radius=2,
                     max_fraction=0.5):
    """Progressively erode the first/last Z-slices of a binary volume.

    The outermost active slice receives ``max_iters`` erosion iterations; each
    successive slice inward receives one fewer, tapering the cell ends so they
    look rounded rather than flat-cut (confocal stacks rarely capture the true
    top/bottom of a cell, so a flat cap is an artefact of the acquisition).

    Args:
        vol: 3-D bool array (Z, Y, X).
        n_slices: Number of slices at each end to taper.
        max_iters: Erosion iterations applied to the outermost slice.
        disk_radius: Radius of the 2-D disk structuring element (XY pixels).
        max_fraction: Maximum fraction of the cell height that may be tapered
            at each end (0.5 = up to half the slices).

    Returns:
        Tapered volume (same shape, bool).
    """
    from skimage.morphology import disk as _sk_disk

    z_any = vol.any(axis=(1, 2))
    if not z_any.any():
        return vol

    z0 = int(z_any.argmax())
    z1 = int(len(z_any) - z_any[::-1].argmax() - 1)
    nz = z1 - z0 + 1

    n_taper = min(n_slices, max(0, int(nz * max_fraction)))
    if n_taper <= 0 or max_iters <= 0:
        return vol

    vol = vol.copy()
    struct2d = _sk_disk(disk_radius).astype(bool)

    for i in range(n_taper):
        # Linear ramp: outermost gets max_iters, innermost gets 1
        iters = max(1, int(round(max_iters * (n_taper - i) / n_taper)))
        zi_top = z0 + i
        zi_bot = z1 - i
        vol[zi_top] = binary_erosion(vol[zi_top], structure=struct2d,
                                     iterations=iters)
        if zi_top != zi_bot:
            vol[zi_bot] = binary_erosion(vol[zi_bot], structure=struct2d,
                                         iterations=iters)

    return vol


# ── mesh smoothing (pure scipy; replaces pyvista/VTK) ─────────────────────────

def _laplacian_smooth(verts, faces, n_iter=50, relaxation=0.1):
    """Uniform-weight Laplacian smoothing.

    Mirrors ``vtkSmoothPolyDataFilter``: each vertex moves a fraction
    ``relaxation`` of the way toward the centroid of its 1-ring neighbours,
    repeated ``n_iter`` times.
    """
    from scipy.sparse import coo_matrix

    if not n_iter or len(verts) == 0 or len(faces) == 0:
        return verts

    n = len(verts)
    a = faces[:, [0, 1, 2]].ravel()
    b = faces[:, [1, 2, 0]].ravel()
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])

    adj = coo_matrix((np.ones(rows.size, np.float32), (rows, cols)),
                     shape=(n, n)).tocsr()
    adj.data[:] = 1.0            # csr merged duplicate edges; make weights uniform
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    v = verts.astype(np.float64, copy=True)
    for _ in range(int(n_iter)):
        v += relaxation * ((adj @ v) / deg[:, None] - v)

    return v.astype(np.float32)


def _fill_holes_pyvista(verts, faces, hole_size=1e9):
    """Seal small surface holes via PyVista, if it is installed.

    Returns the input unchanged when PyVista is unavailable — marching cubes on
    a padded volume is watertight anyway, so this is a no-op in practice.
    """
    try:
        import pyvista as pv
    except ImportError:
        return verts, faces

    try:
        vtk_faces = np.hstack(
            [np.full((faces.shape[0], 1), 3, faces.dtype), faces]).ravel()
        mesh = pv.PolyData(verts, vtk_faces).fill_holes(hole_size)
        tri = mesh.faces.reshape(-1, 4)
        if tri.size == 0 or not np.all(tri[:, 0] == 3):
            return verts, faces
        return np.asarray(mesh.points, np.float32), tri[:, 1:].astype(np.int64)
    except Exception:
        return verts, faces


# ── core: binary volume → mesh ────────────────────────────────────────────────

def mesh_from_binary(vol_bin,
                     spacing=(1.0, 1.0, 1.0),
                     smooth_iter=50,
                     cap_ends=True,
                     fill_mesh_holes=True,
                     hole_size=1e9,
                     resample_isotropic=True,
                     pre_smooth_sigma=0.5,
                     taper_ends=False,
                     taper_n_slices=4,
                     taper_max_iters=5,
                     taper_disk_radius=2,
                     taper_max_fraction=0.5,
                     flip_y=False,
                     return_origin=False):
    """Build a triangular surface mesh from a binary volume.

    Args:
        vol_bin: 3-D boolean/0-1 array in (Z, Y, X) order.
        spacing: (dz, dy, dx) in physical units (µm).
        smooth_iter: Laplacian smoothing iterations on the mesh.
        cap_ends: If the object touches the first/last Z slice, duplicate that
            slice and close along Z so the cap is solid rather than open.
        fill_mesh_holes: Seal small surface holes (requires PyVista; no-op
            without it).
        hole_size: Maximum hole size for the fill, in mesh units.
        resample_isotropic: Upsample Z to match XY spacing before marching
            cubes, avoiding staircase artefacts at large dz/dxy ratios.
        pre_smooth_sigma: Gaussian sigma (voxels) to anti-alias before
            marching cubes.
        taper_ends: Erode the first/last Z-slices so the cell narrows instead
            of ending in a flat cross-section. See :func:`_apply_end_taper`.
        taper_n_slices, taper_max_iters, taper_disk_radius, taper_max_fraction:
            Taper parameters.
        flip_y: Mirror the volume in Y before meshing. The original renderer
            did this to orient its PyVista camera; leave False for napari.
        return_origin: Also return the (z, y, x) offset, in *input voxel*
            units, that internal padding introduced. Only needed when placing
            the mesh back into a larger volume.

    Returns:
        ``(verts, faces)`` — verts float32 (N, 3) in physical units, ordered
        (z, y, x); faces int64 (M, 3) triangle indices. ``None`` if the volume
        is empty. With ``return_origin=True``: ``(verts, faces, origin_vox)``.
    """
    if vol_bin is None or np.asarray(vol_bin).size == 0:
        return None
    vol = np.asarray(vol_bin).astype(bool)
    if not vol.any():
        return None
    vol = vol[:, ::-1, :].copy() if flip_y else vol.copy()

    dz, dy, dx = (float(s) for s in spacing)
    # Offset of the working array's index 0 relative to the input array's
    # index 0, in input-voxel units. Padding steps below shift it.
    origin_vox = np.zeros(3, dtype=np.float64)

    if taper_ends:
        vol = _apply_end_taper(vol,
                               n_slices=taper_n_slices,
                               max_iters=taper_max_iters,
                               disk_radius=taper_disk_radius,
                               max_fraction=taper_max_fraction)
        if not vol.any():
            return None

    # If the object is cut off by the stack boundary, extend it by one slice so
    # marching cubes closes the surface instead of leaving an open rim.
    if cap_ends:
        if vol[0].any():
            vol = np.concatenate([vol[0:1], vol], axis=0)
            origin_vox[0] -= 1.0
        if vol[-1].any():
            vol = np.concatenate([vol, vol[-1:]], axis=0)
        vol = binary_closing(vol, structure=np.ones((3, 1, 1), dtype=bool),
                             iterations=1)

    # Match the Z grid to XY resolution. Linear interpolation + 0.5 threshold
    # avoids the staircase that a nearest-neighbour upsample would bake in at
    # the ~7x dz/dxy ratio typical of these stacks.
    mc_dz = dz
    if resample_isotropic:
        target_xy = max(dy, dx, 1e-8)
        fz = max(1.0, dz / target_xy)
        if abs(fz - 1.0) > 1e-3 and vol.shape[0] > 1:
            nz_in = vol.shape[0]
            vol_f = zoom(vol.astype(np.float32), zoom=(fz, 1.0, 1.0), order=1)
            vol = vol_f > 0.5
            nz_out = vol.shape[0]
            # Exact index mapping of scipy.ndimage.zoom, so the resampled mesh
            # lands on the same physical Z as the source voxels.
            mc_dz = dz * (nz_in - 1) / max(nz_out - 1, 1)
            if not vol.any():
                return None

    volume = vol.astype(np.float32)
    if pre_smooth_sigma and pre_smooth_sigma > 0:
        volume = gaussian_filter(volume, sigma=float(pre_smooth_sigma))

    if volume.max() < 0.5 or volume.min() > 0.5:
        return None

    verts, faces, _normals, _values = measure.marching_cubes(
        volume, level=0.5, spacing=(mc_dz, dy, dx)
    )
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)

    if fill_mesh_holes:
        verts, faces = _fill_holes_pyvista(verts, faces, hole_size)

    if smooth_iter:
        verts = _laplacian_smooth(verts, faces, n_iter=smooth_iter,
                                  relaxation=0.1)

    # Shift out of the padded working frame back into input-array coordinates.
    verts = verts + (origin_vox * (dz, dy, dx)).astype(np.float32)

    if return_origin:
        return verts, faces, origin_vox
    return verts, faces


# ── whole label volume → per-cell meshes ──────────────────────────────────────

def touches_xy_boundary(mask, min_slices=2):
    """True if the mask touches the XY image border in >= ``min_slices`` slices.

    Cells clipped by the field of view have truncated volumes and render as
    sliced-off blobs, so the original pipeline drops them.
    """
    hits = 0
    for z in range(mask.shape[0]):
        s = mask[z]
        if s[0, :].any() or s[-1, :].any() or s[:, 0].any() or s[:, -1].any():
            hits += 1
            if hits >= min_slices:
                return True
    return False


def label_volume_to_meshes(label_vol,
                           spacing=(1.0, 1.0, 1.0),
                           vol_by_id: Mapping[int, float] | None = None,
                           min_volume_um3: float | None = None,
                           max_volume_um3: float | None = None,
                           skip_boundary_cells: bool = True,
                           cell_ids: Iterable[int] | None = None,
                           crop_pad: int = 2,
                           progress: bool = False,
                           **mesh_kwargs):
    """Mesh every qualifying label in a 3-D label volume.

    Args:
        label_vol: (Z, Y, X) integer array; 0 = background.
        spacing: (dz, dy, dx) in µm.
        vol_by_id: Optional ``cell_id -> volume_um3`` map (e.g. from
            ``cell_statistics.csv``). Required for the volume filters; when
            omitted, volume is computed as voxel count x voxel volume.
        min_volume_um3, max_volume_um3: Volume filters in µm³. ``None``
            disables the bound. The repo config used 100 and 10000.
        skip_boundary_cells: Drop cells clipped by the XY field of view.
        cell_ids: Restrict to these labels instead of all non-zero ones.
        crop_pad: Zero-padding (voxels) around each cell's bounding box.
            Needs to be >= 1 so marching cubes closes the surface in XY.
        progress: Print per-cell progress.
        **mesh_kwargs: Forwarded to :func:`mesh_from_binary`. Anything not
            given falls back to :data:`MESH_DEFAULTS`.

    Returns:
        ``{cell_id: (verts, faces)}`` with verts in µm, (z, y, x) order,
        positioned in the coordinate frame of the full ``label_vol``.
    """
    kw = {**MESH_DEFAULTS, **mesh_kwargs}
    kw.pop("return_origin", None)

    label_vol = np.asarray(label_vol)
    dz, dy, dx = (float(s) for s in spacing)
    voxel_um3 = dz * dy * dx
    pad = max(int(crop_pad), 1)

    if cell_ids is None:
        cell_ids = [int(i) for i in np.unique(label_vol) if i != 0]
    else:
        cell_ids = [int(i) for i in cell_ids]

    # One pass with find_objects is far cheaper than N full-volume comparisons.
    slices = _bounding_boxes(label_vol, cell_ids)

    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cell_id in cell_ids:
        sl = slices.get(cell_id)
        if sl is None:
            continue

        if vol_by_id is not None:
            vol_um3 = float(vol_by_id.get(cell_id, 0.0))
        else:
            vol_um3 = float((label_vol[sl] == cell_id).sum()) * voxel_um3

        if min_volume_um3 is not None and vol_um3 < min_volume_um3:
            continue
        if max_volume_um3 is not None and vol_um3 > max_volume_um3:
            continue

        # Pad the crop so the cell is surrounded by background, except where it
        # genuinely runs into the stack edge — there cap_ends must still fire.
        pads, starts = [], []
        for axis, s in enumerate(sl):
            lo_pad = pad if s.start > 0 else 0
            hi_pad = pad if s.stop < label_vol.shape[axis] else 0
            pads.append((lo_pad, hi_pad))
            starts.append(s.start - lo_pad)

        mask = (label_vol[sl] == cell_id)
        if skip_boundary_cells:
            # Boundary test needs the full-frame mask, not the crop.
            full = np.zeros(label_vol.shape, dtype=bool)
            full[sl] = mask
            if touches_xy_boundary(full):
                del full
                continue
            del full

        mask = np.pad(mask, pads, mode="constant", constant_values=False)

        result = mesh_from_binary(mask, spacing=spacing, **kw)
        if result is None:
            if progress:
                print(f"  [skip] id={cell_id}: empty after preprocessing")
            continue

        verts, faces = result
        verts = verts + np.asarray(starts, np.float32) * np.asarray(
            (dz, dy, dx), np.float32)
        meshes[cell_id] = (verts, faces)

        if progress:
            print(f"  id={cell_id:4d}  {vol_um3:8.0f} µm³  "
                  f"{len(verts):6d} verts  {len(faces):6d} faces")

    return meshes


def _bounding_boxes(label_vol, cell_ids):
    """Map cell_id -> tuple of slices, via a single ``find_objects`` pass."""
    from scipy.ndimage import find_objects

    wanted = set(cell_ids)
    objs = find_objects(label_vol.astype(np.int64))
    return {i + 1: sl for i, sl in enumerate(objs)
            if sl is not None and (i + 1) in wanted}


# ── napari / pyvista adapters ─────────────────────────────────────────────────

def meshes_to_napari_surface(meshes: Mapping[int, tuple], values=None):
    """Concatenate per-cell meshes into one napari surface tuple.

    Cheaper than one layer per cell when you have hundreds of cells.

    Args:
        meshes: ``{cell_id: (verts, faces)}`` from :func:`label_volume_to_meshes`.
        values: Optional ``cell_id -> float`` map used as the per-vertex scalar
            (e.g. volume, for colouring). Defaults to the cell id.

    Returns:
        ``(vertices, faces, values)`` ready for ``viewer.add_surface``.
    """
    all_v, all_f, all_val, offset = [], [], [], 0
    for cell_id, (verts, faces) in meshes.items():
        all_v.append(verts)
        all_f.append(faces + offset)
        val = float(cell_id) if values is None else float(
            values.get(cell_id, cell_id))
        all_val.append(np.full(len(verts), val, np.float32))
        offset += len(verts)

    if not all_v:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.int64),
                np.zeros((0,), np.float32))

    return (np.concatenate(all_v), np.concatenate(all_f),
            np.concatenate(all_val))


def add_meshes_to_napari(viewer, meshes: Mapping[int, tuple],
                         separate_layers: bool = False,
                         values=None, name: str = "macrophages",
                         **layer_kwargs):
    """Add meshes to a napari viewer.

    Vertices are already in physical units, so give the matching image/labels
    layer ``scale=spacing`` and everything lines up in world coordinates.

    Args:
        viewer: A ``napari.Viewer`` (use ``ndisplay=3``).
        meshes: ``{cell_id: (verts, faces)}``.
        separate_layers: One layer per cell (selectable individually) instead
            of a single merged surface.
        values: Optional ``cell_id -> float`` for per-vertex colouring.
        name: Layer name (or prefix, when ``separate_layers``).
        **layer_kwargs: Passed through to ``viewer.add_surface``.

    Returns:
        The layer, or list of layers when ``separate_layers=True``.
    """
    if separate_layers:
        return [
            viewer.add_surface(
                (v, f, np.full(len(v), float(cid), np.float32)),
                name=f"{name}_{cid:04d}", **layer_kwargs)
            for cid, (v, f) in meshes.items()
        ]

    return viewer.add_surface(meshes_to_napari_surface(meshes, values),
                              name=name, **layer_kwargs)


def to_pyvista(verts, faces):
    """Wrap ``(verts, faces)`` as a ``pyvista.PolyData``.

    Note the original renderer fed (z, y, x) vertices straight to PyVista,
    which reads them as (x, y, z). This reverses the axes so the PolyData is
    in true (x, y, z) — orientation will differ from the archived PNGs.
    """
    import pyvista as pv

    vtk_faces = np.hstack(
        [np.full((faces.shape[0], 1), 3, faces.dtype), faces]).ravel()
    return pv.PolyData(np.ascontiguousarray(verts[:, ::-1]), vtk_faces)
