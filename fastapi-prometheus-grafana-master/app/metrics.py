"""Prometheus metrics for ML observability."""

from prometheus_client import Counter, Gauge, Histogram

# Custom Prometheus metrics for ML observability
prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["model_name"]
)

detection_counter = Counter(
    "ml_detections_total", 
    "Total number of objects detected",
    ["class_name", "model_name"]
)

confidence_histogram = Histogram(
    "ml_detection_confidence",
    "Distribution of detection confidence scores",
    ["class_name"],
    buckets=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
)

inference_time_histogram = Histogram(
    "ml_inference_duration_ms",
    "Model inference duration in milliseconds",
    ["model_name"],
    buckets=(10, 25, 50, 100, 200, 500, 1000, 2000, 5000)
)

detections_per_image = Histogram(
    "ml_detections_per_image",
    "Number of detections per image",
    ["model_name"],
    buckets=(0, 1, 2, 5, 10, 20, 50, 100)
)

avg_confidence_gauge = Gauge(
    "ml_avg_confidence",
    "Average confidence score of recent predictions",
    ["model_name"]
)

low_confidence_counter = Counter(
    "ml_low_confidence_detections_total",
    "Count of low confidence detections (< 0.5)",
    ["class_name", "model_name"]
)
