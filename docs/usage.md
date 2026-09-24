# Usage

The plugin exposes four commands under **Plugins → napari-macrophage**:

1. **Load Image + Mask** — pick a multi-channel TIFF and a matching mask.
2. **Load from zarr** — load CD206/DAPI/Masks (and any saved ROIs) from a
   `.zarr` directory.
3. **Annotate & Correct Masks/Boxes** — open the main tools dock.
4. **Run Watershed for All ROIs** — batch watershed over every ROI drawn on
   the ROI layer.

## Typical workflow

1. **Load data** — Plugins → napari-macrophage → Load Image + Mask
2. **Open the tools dock** — Plugins → napari-macrophage → Annotate & Correct
   Masks/Boxes. Each section is collapsed by default; click a header to
   expand it.
3. **Set the voxel size** in the *Voxel Size* section. Morphology metrics
   and isotropic resampling need this.
4. **Segment** — draw an ROI bounding box, adjust the Otsu threshold with
   the slider, run Watershed, and save the result back to the Masks layer.
5. **Clean** — expand the *Cleaning* section and run
   *Clean 3D Macrophages*. Each labelled cell is cleaned independently.
6. **Analyse** — expand *Analysis and Processing* → *Cells Analysis* to
   compute volume, surface area, and sphericity per cell; export to CSV.
7. **Visualise** — expand *3D Visualization* → *Generate 3D* to build a
   smoothed surface mesh for one object in a new napari window.

## Keyboard shortcuts

The tools dock registers these hotkeys on the active viewer:

| Key | Action |
|-----|--------|
| `d` | Delete the selected object on the current slice |
| `Shift + D` | Delete the selected object across every slice |
| `v` | View the selected object in its own layer |
| `i` | Open the Cells Analysis dialog |
| `↑` / `↓` | Step through objects on the current slice |

## Log panel

Every `show_info`, `show_warning`, and `show_error` call is captured by the
scrollable **Log** panel at the bottom of the tools dock. The default napari
notification popups are suppressed while the dock is open so messages don't
overlap the viewer.

## Image conventions

- 3D grayscale: `(Z, Y, X)`
- Multi-channel: `(C, Z, Y, X)` where `C ∈ {2, 5}`. Recognised channel names
  are CD206, DAPI, Collagen, F480.
- Masks: `(Z, Y, X)` `uint8` instance labels.
- Formats: TIFF (`.tif` / `.tiff`) and Zarr directories.
