from car5_autolabel.roi_verification import (
    expand_context_crop,
    is_small_proposal,
    make_context_crop,
    select_verified_detection,
    touches_context_crop_edge,
)
from car5_autolabel.schemas import BoundingBox
from car5_autolabel.tiling import TileWindow


def box(
    label: str,
    score: float,
    xyxy: tuple[float, float, float, float],
) -> BoundingBox:
    return BoundingBox(
        label=label,
        score=score,
        x1=xyxy[0],
        y1=xyxy[1],
        x2=xyxy[2],
        y2=xyxy[3],
    )


def test_context_crop_stays_inside_image() -> None:
    crop = make_context_crop(
        box("Truck", 0.3, (0, 0, 20, 20)),
        image_width=1920,
        image_height=1080,
    )

    assert crop == TileWindow(0, 0, 320, 320)


def test_small_proposal_accepts_low_area_target() -> None:
    assert is_small_proposal(
        box("Truck", 0.3, (100, 100, 130, 125)),
        image_width=1920,
        image_height=1080,
    )


def test_context_crop_expands_and_stays_inside_image() -> None:
    assert expand_context_crop(
        TileWindow(0, 0, 320, 320),
        image_width=1920,
        image_height=1080,
        scale=2,
    ) == TileWindow(0, 0, 640, 640)


def test_internal_crop_edge_fragment_is_flagged() -> None:
    crop = TileWindow(400, 200, 720, 520)
    assert touches_context_crop_edge(
        box("Truck", 0.8, (405, 280, 460, 340)),
        crop,
        image_width=1920,
        image_height=1080,
    )


def test_real_image_edge_is_not_treated_as_crop_fragment() -> None:
    crop = TileWindow(0, 0, 320, 320)
    assert not touches_context_crop_edge(
        box("Truck", 0.8, (0, 80, 50, 140)),
        crop,
        image_width=1920,
        image_height=1080,
    )


def test_crop_can_confirm_and_reclassify_a_proposal() -> None:
    proposal = box("Truck", 0.3, (510, 110, 550, 160))
    crop = TileWindow(500, 100, 820, 420)
    local = [box("Excavator", 0.8, (8, 8, 55, 65))]

    verified = select_verified_detection(
        proposal,
        local,
        crop,
        confirmation_threshold=0.5,
        allow_reclassification=True,
    )

    assert verified is not None
    assert verified.label == "Excavator"
    assert verified.postprocess_reason == "native_crop_reclassified_from_Truck"


def test_crop_does_not_reclassify_by_default() -> None:
    proposal = box("Truck", 0.3, (510, 110, 550, 160))
    crop = TileWindow(500, 100, 820, 420)
    local = [box("Excavator", 0.95, (8, 8, 55, 65))]

    assert (
        select_verified_detection(
            proposal,
            local,
            crop,
            confirmation_threshold=0.5,
        )
        is None
    )


def test_crop_rejects_unrelated_detection() -> None:
    proposal = box("Truck", 0.3, (510, 110, 550, 160))
    crop = TileWindow(500, 100, 820, 420)
    local = [box("Truck", 0.9, (200, 200, 250, 250))]

    assert (
        select_verified_detection(
            proposal,
            local,
            crop,
            confirmation_threshold=0.5,
        )
        is None
    )
