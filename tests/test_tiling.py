from car5_autolabel.schemas import BoundingBox
from car5_autolabel.tiling import (
    TileWindow,
    make_tile_windows,
    map_box_from_tile,
    touches_internal_tile_edge,
)


def make_box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(
        label="Truck",
        score=0.8,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def test_tiles_cover_the_far_image_edges() -> None:
    windows = make_tile_windows(1920, 1080, tile_size=960, overlap=0.2)

    assert TileWindow(960, 120, 1920, 1080) in windows
    assert max(window.right for window in windows) == 1920
    assert max(window.bottom for window in windows) == 1080


def test_full_image_is_not_repeated_as_a_tile() -> None:
    assert make_tile_windows(640, 480, tile_size=960, overlap=0.2) == []


def test_internal_edge_fragment_is_rejected_but_image_edge_is_kept() -> None:
    internal = TileWindow(500, 0, 1460, 960)
    image_edge = TileWindow(0, 0, 960, 960)
    near_left = make_box(1, 200, 30, 260)

    assert touches_internal_tile_edge(
        near_left,
        internal,
        image_width=1920,
        image_height=1080,
        margin_ratio=0.04,
    )
    assert not touches_internal_tile_edge(
        near_left,
        image_edge,
        image_width=1920,
        image_height=1080,
        margin_ratio=0.04,
    )


def test_tile_box_maps_back_to_full_image_coordinates() -> None:
    mapped = map_box_from_tile(make_box(10, 20, 50, 80), TileWindow(500, 100, 1460, 1060))

    assert (mapped.x1, mapped.y1, mapped.x2, mapped.y2) == (510, 120, 550, 180)
    assert mapped.postprocess_reason == "high_resolution_tile_candidate"
