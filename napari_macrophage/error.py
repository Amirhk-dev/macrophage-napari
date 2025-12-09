from napari.utils.notifications import show_warning

def _layer_not_in_viewer_error(layer_name: str):
    msg = f"Layer {layer_name} not in the viewer"
    show_warning(msg)
    print(msg)

def _layers_not_in_viewer_error(viewer, required_layers: list[str]):
    missing_layers = [name for name in required_layers if name not in viewer.layers]
    if missing_layers:
        msg = f"Layers not in the viewer: {', '.join(missing_layers)}"
        show_warning(msg)
        print(msg)
        return True 
    return False