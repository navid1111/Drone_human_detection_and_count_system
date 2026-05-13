"""Pydantic models and response schemas."""

from pydantic import BaseModel


class DetectionOut(BaseModel):
    """Detection response schema."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


class PredictionOut(BaseModel):
    """Prediction response schema."""
    id: int
    created_at: str
    model_name: str
    inference_ms: float
    image_url: str
    annotated_image_url: str
    ground_truth: dict | None
    detections: list[DetectionOut]
