import pytest

pytest.importorskip("napari")

from napari_macrophage.state import DataState, dataState, get_voxel_size_um, set_voxel_size_um


@pytest.fixture(autouse=True)
def _reset_state():
    """Restore the module-level ``dataState`` between tests so they don't leak."""
    original = {
        "cd206_images": dataState.cd206_images,
        "dapi_images": dataState.dapi_images,
        "mask_images": dataState.mask_images,
        "collagen_images": dataState.collagen_images,
        "F480_images": dataState.F480_images,
        "mask_path": dataState.mask_path,
        "file_name": dataState.file_name,
        "voxel_size_um": dataState.voxel_size_um,
    }
    yield
    for k, v in original.items():
        setattr(dataState, k, v)


def test_default_dataclass_has_all_none_fields():
    fresh = DataState()
    assert fresh.cd206_images is None
    assert fresh.dapi_images is None
    assert fresh.mask_images is None
    assert fresh.collagen_images is None
    assert fresh.F480_images is None
    assert fresh.mask_path is None
    assert fresh.file_name is None
    assert fresh.voxel_size_um is None


def test_get_voxel_size_um_is_none_when_unset():
    dataState.voxel_size_um = None
    assert get_voxel_size_um() is None


def test_set_voxel_size_um_stores_zyx_order():
    set_voxel_size_um(voxel_x=0.5, voxel_y=0.4, voxel_z=1.2)
    assert dataState.voxel_size_um == (1.2, 0.4, 0.5)
    assert get_voxel_size_um() == (1.2, 0.4, 0.5)


def test_set_voxel_size_um_overwrites_previous_value():
    set_voxel_size_um(1.0, 1.0, 1.0)
    set_voxel_size_um(voxel_x=2.0, voxel_y=3.0, voxel_z=4.0)
    assert get_voxel_size_um() == (4.0, 3.0, 2.0)
