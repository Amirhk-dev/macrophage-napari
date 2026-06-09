from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float = Field(ge=0.0, le=1.0, description="Detection confidence score")


class SliceDetections(BaseModel):
    z: int = Field(description="Z-slice index")
    boxes: list[BoundingBox]


class DetectionResponse(BaseModel):
    model: str = Field(description="ONNX model filename used for inference")
    confidence_threshold: float
    total_boxes: int
    slices: list[SliceDetections]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
