import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("skimage")

from napari_macrophage.clean3d import (
    clean_mask_per_slice,
    largest_3d_component,
    shrink_all_masks_array,
)


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


def test_clean_mask_per_slice_keep_when_empty_false_empties_slice():
    # A 1-pixel-wide line is entirely removed by erosion_radius=3. With
    # keep_when_empty=False (used by the shrink pipeline), the returned slice
    # must be empty instead of falling back to the original mask.
    line = np.zeros((20, 20), dtype=bool)
    line[10, 2:18] = True
    cleaned = clean_mask_per_slice(
        {5: line},
        erosion_radius=3,
        closing_radius=0,
        gaussian_sigma=0.0,
        keep_when_empty=False,
    )
    assert not cleaned[5].any()


def test_largest_3d_component_padded_closing_preserves_boundary_slices():
    # Two disk slices at the very top and bottom of the cropped volume, with
    # a one-slice gap between them. The internal z-padding must protect the
    # top/bottom slices from the closing's erosion phase, so both survive.
    d = _disk(5)
    masks_by_z = {
        0: d.copy(),
        # slice 1 intentionally missing
        2: d.copy(),
    }
    bridged = largest_3d_component(masks_by_z, closing_radius_3d=1)
    assert 0 in bridged and 2 in bridged
    assert bridged[0].any() and bridged[2].any()


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


# ---------------------------------------------------------------------------
# shrink_all_masks_array
# ---------------------------------------------------------------------------


def _labeled_disk_volume(z_extent: range, radius: int, obj_id: int = 1, shape=(20, 20)) -> np.ndarray:
    """Build a (Z, H, W) uint8 mask stack with a filled disk of ``obj_id`` on each z in ``z_extent``."""
    Z = max(z_extent) + 1
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    disk = (yy - shape[0] // 2) ** 2 + (xx - shape[1] // 2) ** 2 <= radius ** 2
    vol = np.zeros((Z,) + shape, dtype=np.uint8)
    for z in z_extent:
        vol[z][disk] = obj_id
    return vol


def test_shrink_all_masks_array_empty_input_returns_empty():
    masks = np.zeros((3, 10, 10), dtype=np.uint8)
    out = shrink_all_masks_array(masks, erosion_radius=1, closing_radius_3d=0)
    assert out.shape == masks.shape
    assert out.dtype == masks.dtype
    assert not out.any()


def test_shrink_all_masks_array_rejects_non_3d_input():
    with pytest.raises(ValueError):
        shrink_all_masks_array(np.zeros((5, 5), dtype=np.uint8))


def test_shrink_all_masks_array_reduces_total_voxels():
    masks = _labeled_disk_volume(range(3), radius=6, obj_id=1)
    before = int((masks > 0).sum())
    out = shrink_all_masks_array(masks, erosion_radius=2, closing_radius_3d=0)
    after = int((out > 0).sum())
    assert after < before
    assert out.dtype == masks.dtype


def test_shrink_all_masks_array_preserves_object_ids():
    # Two objects with distinct ids should survive erosion=1 and keep their ids.
    masks = np.zeros((3, 30, 30), dtype=np.uint8)
    yy, xx = np.mgrid[:30, :30]
    for z in range(3):
        masks[z][((yy - 8) ** 2 + (xx - 8) ** 2) <= 25] = 1
        masks[z][((yy - 22) ** 2 + (xx - 22) ** 2) <= 25] = 2
    out = shrink_all_masks_array(masks, erosion_radius=1, closing_radius_3d=0)
    ids = sorted(int(v) for v in np.unique(out) if v != 0)
    assert ids == [1, 2]


def test_shrink_all_masks_array_drops_tiny_objects():
    # An object that is fully erased by erosion should not appear in the output.
    masks = np.zeros((3, 20, 20), dtype=np.uint8)
    masks[:, 5:15, 5:15] = 1
    masks[:, 17:18, 17:18] = 2  # 1x1 speck per slice — fully eroded
    out = shrink_all_masks_array(masks, erosion_radius=1, closing_radius_3d=0)
    ids = sorted(int(v) for v in np.unique(out) if v != 0)
    assert ids == [1]


def test_shrink_all_masks_array_keeps_largest_2d_component_per_slice():
    # A dumbbell: two blobs joined by a 1-pixel-wide bridge. Erosion=1 breaks
    # the bridge; the resulting slice must retain only the larger blob.
    from skimage.measure import label as sklabel

    Z, H, W = 3, 20, 40
    masks = np.zeros((Z, H, W), dtype=np.uint8)
    for z in range(Z):
        masks[z, 8:14, 5:20] = 1       # large blob
        masks[z, 9:13, 22:30] = 1      # smaller blob
        masks[z, 10:11, 20:22] = 1     # thin bridge (erodes away)
    out = shrink_all_masks_array(masks, erosion_radius=1, closing_radius_3d=0)
    for z in range(Z):
        n_components = sklabel(out[z] == 1).max()
        assert n_components <= 1, f"expected ≤1 2D component on z={z}, got {n_components}"


def test_shrink_all_masks_array_3d_closing_bridges_disappearing_slice():
    # Fat top and bottom slices with a 1-voxel middle slice that vanishes on
    # erosion. Without 3D closing, the object splits into two CCs and only
    # the larger is kept; with closing=1, the object survives as one CC.
    Z, H, W = 6, 20, 20
    masks = np.zeros((Z, H, W), dtype=np.uint8)
    masks[1, 5:15, 5:15] = 1
    masks[3, 5:15, 5:15] = 1
    masks[2, 9:10, 9:10] = 1  # thin bridge — erodes away

    no_close = shrink_all_masks_array(masks, erosion_radius=1, closing_radius_3d=0)
    with_close = shrink_all_masks_array(masks, erosion_radius=1, closing_radius_3d=1)

    # Without closing: middle slice is empty and one of the two halves is dropped.
    assert not (no_close[2] > 0).any()
    slices_kept_no = [z for z in range(Z) if (no_close[z] > 0).any()]
    assert len(slices_kept_no) == 1

    # With closing: gap bridged; both halves survive as one 3D component.
    slices_kept_wc = [z for z in range(Z) if (with_close[z] > 0).any()]
    assert 1 in slices_kept_wc and 3 in slices_kept_wc
