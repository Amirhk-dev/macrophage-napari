import pytest

pytest.importorskip("napari")

from napari_macrophage.error import _layer_not_in_viewer_error, _layers_not_in_viewer_error


class _FakeViewer:
    """Minimal viewer stub: only exposes a ``layers`` container that supports ``in``."""

    def __init__(self, layer_names):
        self.layers = set(layer_names)


def test_layers_not_in_viewer_returns_false_when_all_present():
    viewer = _FakeViewer(["Masks", "CD206", "DAPI"])
    assert _layers_not_in_viewer_error(viewer, ["Masks", "CD206"]) is False


def test_layers_not_in_viewer_returns_true_when_any_missing():
    viewer = _FakeViewer(["Masks"])
    assert _layers_not_in_viewer_error(viewer, ["Masks", "CD206"]) is True


def test_layers_not_in_viewer_returns_true_when_all_missing():
    viewer = _FakeViewer([])
    assert _layers_not_in_viewer_error(viewer, ["Masks", "CD206"]) is True


def test_layers_not_in_viewer_returns_false_for_empty_required():
    viewer = _FakeViewer([])
    assert _layers_not_in_viewer_error(viewer, []) is False


def test_layer_not_in_viewer_error_does_not_raise():
    # Only verifies the helper runs; the user-visible effect is a napari warning.
    _layer_not_in_viewer_error("Masks")
