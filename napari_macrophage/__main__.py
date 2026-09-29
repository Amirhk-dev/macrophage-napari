"""Launcher for napari-macrophage.

Running ``napari-macrophage`` (installed as a console script by pip) starts
napari with the plugin's *Load Image & Mask* and *Annotate & Correct
Masks/Boxes* docks already open, so a user does not have to walk through the
Plugins menu twice on every session. The regular ``napari`` command is
unaffected.
"""

from __future__ import annotations

import napari

from .build_widgets import edit_overlay_all, make_add_layer_from_tif_widget


def main() -> None:
    """Open napari with the two most-used plugin docks pre-mounted."""
    viewer = napari.Viewer()

    load_widget = make_add_layer_from_tif_widget()
    viewer.window.add_dock_widget(load_widget, name="Load Image & Mask", area="right")

    # edit_overlay_all builds the full Macrophage Tools dock directly (skipping
    # the intermediate one-button widget the menu entry uses).
    edit_overlay_all()

    napari.run()


if __name__ == "__main__":
    main()
