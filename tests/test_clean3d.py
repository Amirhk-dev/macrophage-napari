import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("skimage")

from napari_macrophage.clean3d import clean_mask_per_slice, largest_3d_component


def _disk(radius: int) -> np.ndarray:
    """Return a filled boolean disk of the given radius, centred in a square array."""
    n = 2 * radius + 3
    yy, xx = np.mgrid[:n, :n]
    return (yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= radius ** 2


def test_clean_mask_per_slice_removes_stray_pixel():
    mask = _disk(6).copy()
    mask[0, 0] = True  # stray speck disconnected from the disk
    cleaned = clean_mask_per_slice(
        {0: mask}, erosion_radius=1, closing_radius=1, gaussian_sigma=0.5
    )
    assert 0 in cleaned
    assert not cleaned[0][0, 0]  # stray pixel dropped
    # Bulk of the disk should survive
    assert cleaned[0].sum() > mask.sum() * 0.5


def test_clean_mask_per_slice_keeps_original_when_erosion_empties_mask():
    # A thin 1-pixel-wide line disappears entirely under erosion_radius=3,
    # so the helper should fall back to the original mask on that slice.
    line = np.zeros((20, 20), dtype=bool)
    line[10, 2:18] = True
    cleaned = clean_mask_per_slice(
        {5: line}, erosion_radius=3, closing_radius=0, gaussian_sigma=0.0
    )
    # After the fallback path the mask may be re-processed (closing/smoothing),
    # but it must not be empty for a non-empty input.
    assert cleaned[5].any()


def test_clean_mask_per_slice_returns_boolean_dtype():
    cleaned = clean_mask_per_slice({0: _disk(5)})
    assert cleaned[0].dtype == np.bool_


def test_largest_3d_component_keeps_only_biggest():
    # Two separate blobs in 3D. Big blob 4x4x4 = 64 voxels; small blob 2x2x2 = 8.
    m0 = np.zeros((10, 10), dtype=bool)
    m0[1:5, 1:5] = True   # big
    m0[7:9, 7:9] = True   # small
    masks_by_z = {z: m0.copy() for z in range(4)}
    # Trim small blob to only 2 slices, keep big blob on all 4 slices.
    for z in (2, 3):
        masks_by_z[z][7:9, 7:9] = False
    cleaned = largest_3d_component(masks_by_z)
    # Only the big blob region should remain — small-blob pixels are zero.
    for m in cleaned.values():
        assert not m[7:9, 7:9].any()
        assert m[1:5, 1:5].any()


def test_largest_3d_component_min_voxels_drops_object_entirely():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True  # one voxel per slice, 3 slices → 3 voxels total
    masks_by_z = {0: m.copy(), 1: m.copy(), 2: m.copy()}
    cleaned = largest_3d_component(masks_by_z, min_voxels=100)
    assert cleaned == {}


def test_largest_3d_component_min_voxels_keeps_object_above_threshold():
    m = _disk(4)  # ~49 voxels/slice
    masks_by_z = {0: m.copy(), 1: m.copy(), 2: m.copy()}
    cleaned = largest_3d_component(masks_by_z, min_voxels=10)
    assert len(cleaned) == 3
    for z_mask in cleaned.values():
        assert z_mask.any()


def test_largest_3d_component_empty_input_returns_empty():
    assert largest_3d_component({}) == {}


def test_largest_3d_component_3d_closing_bridges_gap():
    # Two disks separated by a one-slice gap (empty slice between them).
    # Pad with empty slices above/below so 3D closing doesn't erode boundary
    # voxels of the disks themselves.
    d = _disk(5)
    empty = np.zeros_like(d)
    masks_by_z = {
        0: empty,
        1: empty,
        2: d.copy(),
        3: d.copy(),
        # slice 4 intentionally missing (the gap under test)
        5: d.copy(),
        6: d.copy(),
        7: empty,
        8: empty,
    }
    cleaned_no_close = largest_3d_component(masks_by_z, closing_radius_3d=0)
    cleaned_closed = largest_3d_component(masks_by_z, closing_radius_3d=2)
    # Without closing, the two disk-pairs are separate CCs; the largest keeps
    # only one pair (≤ 2 non-empty slices).
    assert len(cleaned_no_close) <= 2
    # With closing, the gap fills so the bridged CC spans all four disk slices.
    assert len(cleaned_closed) >= len(cleaned_no_close) + 2
