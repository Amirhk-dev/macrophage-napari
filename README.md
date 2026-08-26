# napari-macrophage

[![PyPI](https://img.shields.io/pypi/v/napari-macrophage)](https://pypi.org/project/napari-macrophage/)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-macrophage)](https://pypi.org/project/napari-macrophage/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/Amirhk-dev/macrophage-napari/blob/main/LICENSE)
[![napari hub](https://img.shields.io/badge/napari%20hub-napari--macrophage-blue)](https://napari-hub.org/plugins/napari-macrophage)

A napari plugin for interactive 3D macrophage image analysis — mask editing, Otsu/Watershed segmentation, YOLO bounding box export, and morphology analysis.

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="docs/3D_generation.gif" width="500" alt="Demo" /></td>
      <td align="center"><img src="docs/3D_rendered_sample.png" width="300" alt="3D rendered macrophage" /></td>
    </tr>
    <tr>
      <td align="center"><em>3D segmentation of macrophages overlaid with the volume</em></td>
      <td align="center"><em>3D rendering of a single macrophage</em></td>
    </tr>
  </table>
</div>

## Features

- Load multi-channel TIFF/Zarr images (CD206, DAPI, Collagen, F480) and 3D instance masks
- Click-to-select objects; delete per-slice or globally; rename, renumber IDs
- Draw ROI → Otsu preview (adjustable threshold) → optional Watershed → save 3D mask
- ONNX-based automatic macrophage detection (CD206 + DAPI)
- Annotate and export/import bounding boxes in YOLO `.txt` format
- Per-object morphology analysis: volume, surface area, sphericity → CSV export
- Isotropic resampling of image and mask
- 3D rendering of individual macrophages (smoothed surface mesh, adjustable shading, black/white background, PNG screenshot, mesh export to STL/OBJ/PLY)

## Installation

**With uv (recommended)**
```bash
uv sync                       # core deps
uv sync --extra detection     # + onnxruntime for ONNX detection
uv run napari
```

**With pip**
```bash
pip install napari-macrophage
napari
```

**Development**
```bash
pip install -e .
napari
```

## Usage

1. **Load data** — Plugins → napari-macrophage → Load Image + Mask
2. **Edit masks** — Plugins → napari-macrophage → Edit CD206 + DAPI + Masks
3. **Segment** — Draw ROI bbox → Otsu preview → Save or Run Watershed
4. **Detect** — Run ONNX detection on CD206 + DAPI slices
5. **Render 3D** — In the *3D Visualization* panel, enter an Object ID and click *Generate 3D* to open the macrophage in a new window; save a PNG or export the mesh (STL/OBJ/PLY) from that window
6. **Export** — YOLO `.txt` bounding boxes or morphology `.csv`

Input shape: `(Z, Y, X)` for grayscale, `(C, Z, Y, X)` for multi-channel (C ∈ {2, 5}).

## Companion pipeline

For fully automated end-to-end segmentation (YOLO + SAM2 + Cellpose), see:
**[macrophage-image-processor](https://github.com/Amirhk-dev/macrophage-image-processor)**
