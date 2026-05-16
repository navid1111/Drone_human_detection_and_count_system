"""Detection service for ONNX inference."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from config import CONF_THRESHOLD, INPUT_HW, IOU_THRESHOLD, MODEL_PATH
from utils import DetectionResult, postprocess, preprocess

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


class DetectionService:
    """Service for running ONNX model inference."""
    
    def __init__(self):
        self.onnx_session: ort.InferenceSession | None = None
        self.sahi_model = None
        self.input_name: str | None = None
        self.input_hw: tuple[int, int] = INPUT_HW
        self.model_name: str = ""

    def initialize(self, model_path: str = MODEL_PATH) -> None:
        """Initialize the ONNX model."""
        if not Path(model_path).exists():
            raise RuntimeError(f"ONNX model not found at: {model_path}")

        self.onnx_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        input_info = self.onnx_session.get_inputs()[0]
        self.input_name = input_info.name
        self.model_name = Path(model_path).name

        shape = input_info.shape
        if isinstance(shape, list) and len(shape) >= 4:
            h = shape[2] if isinstance(shape[2], int) else 640
            w = shape[3] if isinstance(shape[3], int) else 640
            self.input_hw = (int(h), int(w))
            
        if SAHI_AVAILABLE:
            try:
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=model_path,
                    device="cpu",
                )
            except Exception as e:
                print(f"Warning: SAHI model failed to load. SAHI inference will be unavailable. Error: {e}")

    def is_ready(self) -> bool:
        """Check if the model is initialized."""
        return self.onnx_session is not None and self.input_name is not None

    def infer(
        self,
        image_np: np.ndarray,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        use_sahi: bool = False,
    ) -> tuple[list[DetectionResult], float]:
        """
        Run inference on the image.
        
        Returns:
            Tuple of (detections, inference_time_ms)
        """
        if not self.is_ready():
            raise RuntimeError("Model is not initialized")

        if use_sahi and self.sahi_model is not None:
            start = time.perf_counter()
            results = get_sliced_prediction(
                image_np,
                self.sahi_model,
                slice_height=512,
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            inference_ms = (time.perf_counter() - start) * 1000.0
            
            detections = []
            for obj in results.object_prediction_list:
                if obj.score.value >= conf_threshold:
                    detections.append(
                        DetectionResult(
                            class_id=obj.category.id,
                            confidence=obj.score.value,
                            box_xyxy=[obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy]
                        )
                    )
            return detections, inference_ms

        # Preprocess
        model_input = preprocess(image_np, self.input_hw)

        # Inference
        start = time.perf_counter()
        outputs = self.onnx_session.run(None, {self.input_name: model_input})
        inference_ms = (time.perf_counter() - start) * 1000.0

        # Postprocess
        orig_h, orig_w = image_np.shape[:2]
        detections = postprocess(
            outputs[0],
            orig_w=orig_w,
            orig_h=orig_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            input_hw=self.input_hw,
        )

        return detections, inference_ms
