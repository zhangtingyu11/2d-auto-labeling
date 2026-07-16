from car5_autolabel.integrations.label_studio import box_to_label_studio
from car5_autolabel.schemas import BoundingBox


def test_box_to_label_studio_percentage_coordinates() -> None:
    box = BoundingBox(x1=192, y1=108, x2=960, y2=540, label="Truck", score=0.9)

    result = box_to_label_studio(box, width=1920, height=1080)

    assert result["value"] == {
        "x": 10.0,
        "y": 10.0,
        "width": 40.0,
        "height": 40.0,
        "rotation": 0,
        "rectanglelabels": ["Truck"],
    }
