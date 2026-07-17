import re
from datetime import datetime
from pathlib import Path

CAMERA_ORDER = (
    "front",
    "front_left",
    "front_right",
    "back",
    "back_left",
    "back_right",
)
CAMERA_INDEX = {camera: index for index, camera in enumerate(CAMERA_ORDER)}

ACTIVE_CLASSES = (
    "Car",
    "Truck",
    "Bulldozer",
    "Excavator",
    "WaterTruck",
    "Sign",
)
CLASS_INDEX = {label: index for index, label in enumerate(ACTIVE_CLASSES)}

EXPECTED_IMAGE_WIDTH = 1920
EXPECTED_IMAGE_HEIGHT = 1080
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}$")


def parse_capture_timestamp(value: str | Path) -> datetime:
    """Parse a car5 filename stem without relying on lexical ordering."""
    stem = Path(value).stem if isinstance(value, Path) else value
    if not TIMESTAMP_PATTERN.fullmatch(stem):
        raise ValueError(f"invalid car5 timestamp stem: {stem!r}")
    return datetime.strptime(stem, TIMESTAMP_FORMAT)
