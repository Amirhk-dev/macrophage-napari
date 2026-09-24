import pytest

pytest.importorskip("napari")

from napari_macrophage.bbox import _nms_drop_smaller_on_overlap


def test_nms_empty_input_returns_empty():
    assert _nms_drop_smaller_on_overlap([]) == []


def test_nms_single_box_is_kept():
    box = (0.0, 0.0, 10.0, 10.0)
    assert _nms_drop_smaller_on_overlap([box]) == [box]


def test_nms_non_overlapping_boxes_all_kept():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    kept = _nms_drop_smaller_on_overlap([a, b])
    assert set(kept) == {a, b}


def test_nms_drops_smaller_when_iou_above_threshold():
    # Small box fully contained in larger box → IoU = small_area / large_area
    # small = 4*4 = 16, large = 10*10 = 100 → IoU = 0.16 → below default 0.2 → both kept.
    large = (0.0, 0.0, 10.0, 10.0)
    small_contained_below = (3.0, 3.0, 7.0, 7.0)  # 4x4, IoU=0.16
    kept = _nms_drop_smaller_on_overlap([large, small_contained_below])
    assert set(kept) == {large, small_contained_below}


def test_nms_drops_smaller_when_iou_strictly_above_threshold():
    # Bigger contained box: 5x5 = 25, large 10x10 = 100 → IoU = 0.25 > 0.2 → drop smaller
    large = (0.0, 0.0, 10.0, 10.0)
    small_contained_above = (3.0, 3.0, 8.0, 8.0)  # 5x5, IoU=0.25
    kept = _nms_drop_smaller_on_overlap([large, small_contained_above])
    assert kept == [large]


def test_nms_uses_area_not_input_order():
    # Feed the smaller box first; NMS should still keep the larger one.
    large = (0.0, 0.0, 10.0, 10.0)
    small = (3.0, 3.0, 8.0, 8.0)  # IoU with large = 0.25
    kept = _nms_drop_smaller_on_overlap([small, large])
    assert kept == [large]


def test_nms_custom_threshold_zero_keeps_only_non_overlapping():
    # With iou_threshold=0.0, any positive overlap eliminates the smaller box.
    large = (0.0, 0.0, 10.0, 10.0)
    tiny = (9.0, 9.0, 11.0, 11.0)  # 1x1 overlap → IoU ~ 1/104
    kept = _nms_drop_smaller_on_overlap([large, tiny], iou_threshold=0.0)
    assert kept == [large]


def test_nms_chain_of_overlaps_keeps_only_the_largest():
    # Three boxes each pairwise overlapping > 0.2 with the largest.
    biggest = (0.0, 0.0, 10.0, 10.0)     # area 100
    middle = (1.0, 1.0, 9.0, 9.0)        # area 64, contained ⇒ IoU=0.64
    smallest = (2.0, 2.0, 8.0, 8.0)      # area 36, contained ⇒ IoU=0.36 with biggest
    kept = _nms_drop_smaller_on_overlap([middle, smallest, biggest])
    assert kept == [biggest]
