from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)
    label: str
    score: float = Field(ge=0, le=1)
    track_id: str | None = None

    @model_validator(mode="after")
    def validate_corners(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("x2/y2 must be greater than x1/y1")
        return self


class ImagePrediction(BaseModel):
    image_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    model_version: str
    boxes: list[BoundingBox] = Field(default_factory=list)
