"""Database service for predictions and detections."""

import json
from pathlib import Path

from .database import Detection, Prediction, SessionLocal
from .schemas import DetectionOut, PredictionOut
from .utils import DetectionResult, get_class_name


def save_prediction(
    image_path: str,
    annotated_image_path: str,
    model_name: str,
    inference_ms: float,
    detections: list[DetectionResult],
    ground_truth_json: str | None = None,
) -> Prediction:
    """Save prediction and detections to the database."""
    session = SessionLocal()
    try:
        prediction = Prediction(
            image_path=image_path,
            annotated_image_path=annotated_image_path,
            model_name=model_name,
            inference_ms=inference_ms,
            ground_truth_json=ground_truth_json,
        )
        session.add(prediction)
        session.flush()

        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            session.add(
                Detection(
                    prediction_id=prediction.id,
                    class_id=det.class_id,
                    class_name=get_class_name(det.class_id),
                    confidence=det.confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        session.commit()
        session.refresh(prediction)
        _ = prediction.detections
        return prediction
    finally:
        session.close()


def get_predictions(limit: int = 20) -> list[PredictionOut]:
    """Get recent predictions from the database."""
    session = SessionLocal()
    try:
        rows = (
            session.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
        for row in rows:
            _ = row.detections
        return [_prediction_to_response(row) for row in rows]
    finally:
        session.close()


def get_prediction(prediction_id: int) -> PredictionOut | None:
    """Get a specific prediction by ID."""
    session = SessionLocal()
    try:
        row = session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if row is None:
            return None
        _ = row.detections
        return _prediction_to_response(row)
    finally:
        session.close()


def _prediction_to_response(pred: Prediction) -> PredictionOut:
    """Convert a Prediction object to a PredictionOut response."""
    ground_truth = None
    if pred.ground_truth_json:
        try:
            ground_truth = json.loads(pred.ground_truth_json)
        except json.JSONDecodeError:
            ground_truth = {"raw": pred.ground_truth_json}

    detections = [
        DetectionOut(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox_xyxy=[d.x1, d.y1, d.x2, d.y2],
        )
        for d in pred.detections
    ]

    return PredictionOut(
        id=pred.id,
        created_at=pred.created_at.isoformat(),
        model_name=pred.model_name,
        inference_ms=pred.inference_ms,
        image_url=f"/uploads/{Path(pred.image_path).name}",
        annotated_image_url=f"/uploads/{Path(pred.annotated_image_path).name}",
        ground_truth=ground_truth,
        detections=detections,
    )
