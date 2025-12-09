import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")
pytest.importorskip("skimage")

from napari_macrophage.analysis import _compute_volume, _compute_sphericity

# run pytest in tests/
def test_compute_volume_unit_cube_voxel_size_one():
    # Test on a 3x3x3 solid cube labeled as 1 inside a larger array
    mask = np.zeros((6, 6, 6), dtype=np.int32)
    mask[1:4, 1:4, 1:4] = 1  # 3*3*3 = 27 voxels
    result = _compute_volume(1.0, 1.0, 1.0, mask)
    assert result[1] == pytest.approx(27.0)


def test_compute_volume_multiple_labels_anisotropic_voxels():
    # Test on a 3x3x3 array with multiple labels and anisotropic voxel sizes
    # Construct small 3x3x3 array with:
    # - label 1 count = 10 voxels
    # - label 2 count = 4 voxels
    mask = np.zeros((3, 3, 3), dtype=np.uint8)

    # label 1: an 8-voxel block + 2 additional voxels
    mask[0:2, 0:2, 0:2] = 1
    mask[2, 2, 2] = 1   
    mask[2, 2, 1] = 1      

    # label 2: a 2x2 square at z=2
    mask[0, 0:2, 2] = 2
    mask[1, 0:2, 2] = 2   

    # Anisotropic voxel size
    vx, vy, vz = 0.5, 0.2, 1.5
    voxel_volume = vx * vy * vz  # 0.15

    result = _compute_volume(vx, vy, vz, mask)
    assert set(result.keys()) == {1, 2}
    assert np.isclose(result[1], 10 * voxel_volume)
    assert np.isclose(result[2], 4 * voxel_volume)


def test_compute_volume_all_zeros_returns_empty():
    # Test on an all-zero mask returns empty dictionary
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    result = _compute_volume(1.0, 1.0, 1.0, mask)
    assert result == {}


def test_compute_volume_non_contiguous_labels_and_types():
    # Test on non-contiguous labels and different data types
    mask = np.zeros((3, 3, 3), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    mask[1, 1, 1] = 5.0
    result = _compute_volume(1.0, 1.0, 1.0, mask)
    assert set(result.keys()) == {1, 5}
    assert 0 not in result # background (0) is not included


def test_compute_sphericity_unit_sphere_close_to_one():
    # for isotropic voxels: unit sphere sphericity ≈ 1
    print("Test on a unit sphere with isotropic voxels")
    r = 50
    size = 2 * r + 5
    cz = cy = cx = r + 2
    z = np.arange(size)[:, None, None]
    y = np.arange(size)[None, :, None]
    x = np.arange(size)[None, None, :]
    euclidean_dist = (z-cz) ** 2 + (y-cy) ** 2 + (x-cx) ** 2

    mask = np.zeros((size, size, size), dtype=np.uint8)
    mask[euclidean_dist <= r ** 2] = 1

    s, _ = _compute_sphericity(1, 1, 1, mask)
    sph = s[1]
    print(s, sph)
    assert 1 in s
    assert np.isfinite(s[1])
    assert sph > 0.9


def test_compute_sphericity_elongated_object():
    mask = np.zeros((200, 80, 80), dtype=np.uint8)
    mask[50:150, 24:60, 24:60] = 1
    s, _ = _compute_sphericity(1.0, 1.0, 1.0, mask)
    sph = s[1]
    print(s, sph)
    assert np.isfinite(sph)
    assert sph < 0.3
