# Usage

The plugin exposes four commands under **Plugins → napari-macrophage**:

1. **Load Image & Mask** — pick a multi-channel TIFF and a matching mask.
2. **Load from zarr** — load CD206/DAPI/Masks (and any saved ROIs) from a
   `.zarr` directory.
3. **Annotate & Correct Masks/Boxes** — open the main tools dock.
4. **Run Watershed for All ROIs** — batch watershed over every ROI drawn on
   the ROI layer.

## Quick launch

Instead of clicking through the Plugins menu, run the console command
installed by pip:

```bash
napari-macrophage
```

This opens napari with the *Load Image & Mask* dock and the full
*Macrophage Tools* dock already mounted on the right, stacked vertically.
The plain `napari` command still works and is unchanged.

## Typical workflow

1. **Load data** — Plugins → napari-macrophage → Load Image & Mask
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

## Shrink all masks (signal-based)

The *Objects* section has a **Shrink all masks (preview)** action that fits
every labelled object in the *Masks* layer to the actual **CD206** signal
(and, optionally, DAPI). It is the batch version of the single-object
**Shrink Mask** button — the same per-slice method, applied to every object
at once. Nothing is written until you press **Accept**.

Per slice, for each object, it:

1. Normalises CD206 (and DAPI, if used) inside the mask.
2. Computes an Otsu threshold on the values *inside the mask only*, so the
   threshold separates bright cell signal from dim regions.
3. Keeps only pixels above the threshold — unioned with DAPI's own
   above-threshold pixels if *Use DAPI* is on.
4. Retains the largest 2D connected component and fills interior holes.
5. Smooths the boundary with a Gaussian of *Smoothing sigma*.
6. Intersects with the original mask (the result can only shrink).

Across Z, an optional 3D closing bridges any slices that vanished during
the fit, and the largest 3D component is kept.

### Parameters

Click the **? Help — what do these parameters do?** button under the
preview widget for a rich-text popup with the full explanation. In short:

- **Use DAPI** — if on and a DAPI channel is loaded, the mask is kept where
  either CD206 *or* DAPI is bright. Turn off to fit strictly to CD206.
- **Smoothing sigma** — boundary smoothness. Low (0–1) follows the signal
  closely with jagged edges; 2 (default) gives a smooth cell-like boundary;
  high (4+) is very rounded and may bleed past subtle details.
- **3D closing radius** — bridges Z-gaps if the shrink removes a slice
  entirely. `0` keeps only the largest 3D piece; `1` bridges one-slice gaps
  (recommended for thin cells); `2–3` bridges larger gaps but can merge
  nearby objects.

### Workflow

1. Click **Shrink all masks (preview)**. The original *Masks* layer is
   hidden and a new **Masks (shrink preview)** layer appears at 70 % opacity
   so you can visually compare. A log message reports how many voxels were
   removed. Large volumes can take a few seconds — the button shows a wait
   message while the compute runs.
2. Choose an outcome:
   - **Accept** commits the preview into *Masks* and keeps the pre-shrink
     state in memory so you can undo it later.
   - **Cancel** removes the preview and re-shows the original masks
     unchanged. Nothing was modified.
   - **Undo Shrink** (after an Accept) restores the *Masks* layer to its
     pre-shrink state.

### Notes and edge cases

- Requires the **CD206** channel to be loaded (same as *Shrink Mask*).
- On slices where CD206 is completely constant (no threshold can be
  computed), the original mask for that slice is kept so no false Z-gap is
  introduced.
- If shrinking splits an object on a slice into several 2D components, only
  the largest 2D component is kept.
- If an intermediate slice vanishes and *3D closing radius* > 0, the object
  is bridged back into a single 3D component. Otherwise, only the largest
  3D component across the remaining slices is kept.

## Triangulation overlay in *Generate 3D*

The 3D window opened by *Generate 3D* now has a **Show Triangulation
(wireframe)** checkbox and a **Line width** spin box in its *Publication
Tools* dock. Ticking the checkbox draws black wireframe edges on top of the
mesh so the marching-cubes triangulation is visible; the line-width control
adjusts thickness. The overlay is off by default and independent of the
solid mesh — you can toggle it any time to switch between smooth and
wireframe views for figures.

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
