"""Small helpers for reporting missing viewer layers to the user."""

from napari.utils.notifications import show_warning


def _layer_not_in_viewer_error(layer_name: str):
    """Show a warning that the named layer is missing from the viewer."""
    msg = f"Layer {layer_name} not in the viewer"
    show_warning(msg)


def _layers_not_in_viewer_error(viewer, required_layers: list[str]) -> bool:
    """Warn about any missing layers and report whether the check failed.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer whose layer list is inspected.
    required_layers : list of str
        Layer names that must all be present.

    Returns
    -------
    bool
        ``True`` if at least one required layer is missing (a warning has been
        shown), ``False`` if every required layer is present.
    """
    missing_layers = [name for name in required_layers if name not in viewer.layers]
    if missing_layers:
        msg = f"Layers not in the viewer: {', '.join(missing_layers)}"
        show_warning(msg)
        return True
    return False
