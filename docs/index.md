# napari-macrophage

A [napari](https://napari.org/) plugin for interactive 3D microscopy image
analysis of macrophage cells. Provides tools for image loading, mask editing,
Otsu + Watershed segmentation, YOLO-format bounding-box annotation, and
morphological analysis (volume, surface area, sphericity).

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
api
```

## Highlights

- Load multi-channel 3D TIFFs (CD206, DAPI, Collagen, F480) or Zarr datasets
- Draw ROIs and run interactive Otsu → Watershed segmentation
- Per-object 3D cleaning (erosion, closing, largest component, min-voxel filter)
- ONNX object detection on CD206 + DAPI slices
- YOLO bounding-box import/export
- Per-cell morphology CSV export
- Standalone 3D mesh viewer with STL/OBJ/PLY export
- In-plugin log panel that captures every notification

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
