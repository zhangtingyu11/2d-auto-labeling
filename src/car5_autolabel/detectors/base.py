from abc import ABC, abstractmethod
from pathlib import Path

from car5_autolabel.schemas import ImagePrediction


class Detector(ABC):
    """Model-independent detector contract."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: Path) -> ImagePrediction:
        raise NotImplementedError
