from car5_autolabel.policies.size import filter_minimum_long_side, long_side_px
from car5_autolabel.schemas import BoundingBox


def make_box(width: float, height: float) -> BoundingBox:
    return BoundingBox(
        x1=10,
        y1=20,
        x2=10 + width,
        y2=20 + height,
        label="Truck",
        score=0.9,
    )


def test_long_side_uses_rounded_pixel_geometry() -> None:
    assert long_side_px(make_box(69.6, 30)) == 70


def test_70px_boundary_is_inclusive() -> None:
    kept, below_policy = filter_minimum_long_side(
        [make_box(70, 20), make_box(69, 60)],
        minimum_px=70,
    )

    assert [long_side_px(box) for box in kept] == [70]
    assert [long_side_px(box) for box in below_policy] == [69]


def test_negative_boundary_is_rejected() -> None:
    try:
        filter_minimum_long_side([], minimum_px=-1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative minimum must fail")
