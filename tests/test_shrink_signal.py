"""Tests for the signal-based Shrink-all-masks primitives."""

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("skimage")

from napari_macrophage.edit_mask_image import (
    _norm_channel,
    _shrink_slice_by_signal_normalized,
    shrink_all_masks_by_signal_array,
)


def _bright_cell_slice(H: int = 40, W: int = 40, mask_radius: int = 12, bright_radius: int = 7):
    """Return (obj_mask, cd206) where CD206 is bright only inside an inner disk.

    The mask covers a larger disk than the bright signal so shrinking must
    tighten the mask down toward the bright core.
    """
    yy, xx = np.mgrid[:H, :W]
    cy, cx = H // 2, W // 2
    obj_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= mask_radius ** 2
    cd206 = np.zeros((H, W), dtype=np.float32)
    inside_bright = (yy - cy) ** 2 + (xx - cx) ** 2 <= bright_radius ** 2
    cd206[inside_bright] = 1.0
    cd206[obj_mask & ~inside_bright] = 0.05  # dim mask periphery
    return obj_mask, cd206


# ---------------------------------------------------------------------------
# _norm_channel
# ---------------------------------------------------------------------------


def test_norm_channel_maps_to_unit_interval():
    a = np.array([[10.0, 30.0], [20.0, 40.0]])
    n = _norm_channel(a)
    assert n is not None
    assert n.min() == 0.0 and n.max() == 1.0


def test_norm_channel_returns_none_when_constant():
    assert _norm_channel(np.full((5, 5), 7.0)) is None


# ---------------------------------------------------------------------------
# _shrink_slice_by_signal_normalized
# ---------------------------------------------------------------------------


def test_shrink_slice_by_signal_tightens_to_bright_region():
    obj_mask, cd206 = _bright_cell_slice()
    cd206_norm = _norm_channel(cd206)
    shrunk = _shrink_slice_by_signal_normalized(obj_mask, cd206_norm, None, gaussian_sigma=1.0)
    assert shrunk is not None
    assert shrunk.dtype == np.bool_
    assert shrunk.sum() < obj_mask.sum(), "shrink should reduce the mask"
    # Only pixels inside the original mask remain.
    assert not (shrunk & ~obj_mask).any()


def test_shrink_slice_by_signal_returns_none_on_empty_mask():
    _, cd206 = _bright_cell_slice()
    cd206_norm = _norm_channel(cd206)
    empty = np.zeros_like(cd206, dtype=bool)
    assert _shrink_slice_by_signal_normalized(empty, cd206_norm, None) is None


def test_shrink_slice_by_signal_returns_none_when_cd206_norm_missing():
    obj_mask, _ = _bright_cell_slice()
    assert _shrink_slice_by_signal_normalized(obj_mask, None, None) is None


def test_shrink_slice_by_signal_dapi_union_expands_kept_region():
    # CD206 bright on the left half of the mask; DAPI bright on the right.
    # With DAPI the union should keep more pixels than CD206 alone.
    H = W = 30
    yy, xx = np.mgrid[:H, :W]
    obj_mask = (yy >= 5) & (yy < 25) & (xx >= 5) & (xx < 25)
    cd206 = np.zeros((H, W), dtype=np.float32)
    cd206[obj_mask & (xx < 15)] = 1.0
    dapi = np.zeros((H, W), dtype=np.float32)
    dapi[obj_mask & (xx >= 15)] = 1.0

    cd206_norm = _norm_channel(cd206)
    dapi_norm = _norm_channel(dapi)

    cd206_only = _shrink_slice_by_signal_normalized(obj_mask, cd206_norm, None, gaussian_sigma=0.0)
    with_dapi = _shrink_slice_by_signal_normalized(obj_mask, cd206_norm, dapi_norm, gaussian_sigma=0.0)
    assert cd206_only is not None and with_dapi is not None
    assert with_dapi.sum() > cd206_only.sum()


# ---------------------------------------------------------------------------
# shrink_all_masks_by_signal_array
# ---------------------------------------------------------------------------


def _bright_cell_volume(Z=3, obj_id=1, mask_radius=12, bright_radius=7):
    """Build a (Z, H, W) mask volume and matching CD206 image with a bright core."""
    H = W = 40
    masks = np.zeros((Z, H, W), dtype=np.uint8)
    cd206 = np.zeros((Z, H, W), dtype=np.float32)
    for z in range(Z):
        obj_mask, cd = _bright_cell_slice(H, W, mask_radius, bright_radius)
        masks[z][obj_mask] = obj_id
        cd206[z] = cd
    return masks, cd206


def test_batch_shrink_empty_masks_returns_zero_volume():
    masks = np.zeros((3, 10, 10), dtype=np.uint8)
    cd206 = np.random.default_rng(0).random(masks.shape).astype(np.float32)
    out = shrink_all_masks_by_signal_array(masks, cd206)
    assert out.shape == masks.shape
    assert out.dtype == masks.dtype
    assert not out.any()


def test_batch_shrink_rejects_shape_mismatch():
    masks = np.zeros((3, 10, 10), dtype=np.uint8)
    cd206 = np.zeros((3, 12, 12), dtype=np.float32)
    with pytest.raises(ValueError):
        shrink_all_masks_by_signal_array(masks, cd206)


def test_batch_shrink_rejects_non_3d_masks():
    with pytest.raises(ValueError):
        shrink_all_masks_by_signal_array(
            np.zeros((5, 5), dtype=np.uint8),
            np.zeros((5, 5), dtype=np.float32),
        )


def test_batch_shrink_reduces_total_voxels_toward_bright_core():
    masks, cd206 = _bright_cell_volume(Z=3)
    before = int((masks > 0).sum())
    out = shrink_all_masks_by_signal_array(masks, cd206, dapi=None, use_dapi=False, gaussian_sigma=1.0)
    after = int((out > 0).sum())
    assert after < before
    assert out.dtype == masks.dtype
    # Every remaining voxel must have been inside the original mask.
    assert ((out > 0) & (masks == 0)).sum() == 0


def test_batch_shrink_preserves_object_ids():
    # Two objects with different ids, both with a bright core.
    H = W = 60
    masks = np.zeros((2, H, W), dtype=np.uint8)
    cd206 = np.zeros((2, H, W), dtype=np.float32)
    yy, xx = np.mgrid[:H, :W]
    for z in range(2):
        m1 = (yy - 15) ** 2 + (xx - 15) ** 2 <= 100
        b1 = (yy - 15) ** 2 + (xx - 15) ** 2 <= 25
        m2 = (yy - 45) ** 2 + (xx - 45) ** 2 <= 100
        b2 = (yy - 45) ** 2 + (xx - 45) ** 2 <= 25
        masks[z][m1] = 1
        masks[z][m2] = 2
        cd206[z][b1 | b2] = 1.0
    out = shrink_all_masks_by_signal_array(masks, cd206, dapi=None, use_dapi=False, gaussian_sigma=0.0)
    ids = sorted(int(v) for v in np.unique(out) if v != 0)
    assert ids == [1, 2]


def test_batch_shrink_degenerate_signal_slice_keeps_original():
    # Slice 0 CD206 is bright core → shrinks. Slice 1 CD206 is constant →
    # signal is degenerate, so the original mask must be preserved.
    Z = 2
    H = W = 30
    masks = np.zeros((Z, H, W), dtype=np.uint8)
    cd206 = np.zeros((Z, H, W), dtype=np.float32)

    yy, xx = np.mgrid[:H, :W]
    disk = (yy - 15) ** 2 + (xx - 15) ** 2 <= 100
    bright = (yy - 15) ** 2 + (xx - 15) ** 2 <= 25
    masks[0][disk] = 1
    masks[1][disk] = 1
    cd206[0][bright] = 1.0
    cd206[1][:] = 0.5  # constant everywhere → degenerate

    out = shrink_all_masks_by_signal_array(masks, cd206, dapi=None, use_dapi=False, gaussian_sigma=0.0)

    # Slice 0 was shrunk; slice 1 must still match the original mask.
    assert (out[0] > 0).sum() < (masks[0] > 0).sum()
    assert np.array_equal((out[1] > 0), disk)


def test_batch_shrink_use_dapi_expands_kept_region():
    # A mask where CD206 covers the left half and DAPI the right.
    # With use_dapi=True, more pixels should remain than with use_dapi=False.
    Z = 1
    H = W = 40
    yy, xx = np.mgrid[:H, :W]
    disk = (yy - 20) ** 2 + (xx - 20) ** 2 <= 100
    masks = np.zeros((Z, H, W), dtype=np.uint8)
    masks[0][disk] = 1

    cd206 = np.zeros((Z, H, W), dtype=np.float32)
    dapi = np.zeros((Z, H, W), dtype=np.float32)
    cd206[0][disk & (xx < 20)] = 1.0
    dapi[0][disk & (xx >= 20)] = 1.0

    cd206_only = shrink_all_masks_by_signal_array(masks, cd206, dapi=dapi, use_dapi=False, gaussian_sigma=0.0)
    with_dapi = shrink_all_masks_by_signal_array(masks, cd206, dapi=dapi, use_dapi=True, gaussian_sigma=0.0)
    assert (with_dapi > 0).sum() > (cd206_only > 0).sum()
