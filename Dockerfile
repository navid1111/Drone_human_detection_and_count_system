ARG AIRFLOW_BASE_IMAGE=apache/airflow:3.1.6
FROM ${AIRFLOW_BASE_IMAGE}

ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG ULTRALYTICS_VERSION=8.4.18
ARG OPENCV_HEADLESS_VERSION=4.13.0.92
ARG PYYAML_VERSION=6.0.3
ARG PILLOW_VERSION=12.1.1
ARG MATPLOTLIB_VERSION=3.10.8
ARG SCIPY_VERSION=1.17.1
ARG ONNX_VERSION=1.17.0
ARG ONNXRUNTIME_VERSION=1.20.1
ARG INSTALL_WANDB=true

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*
USER airflow

# Install ML/DL dependencies (headless OpenCV for container use)
# Upgrade numpy+pandas first to avoid binary incompatibility with base image
RUN pip install --no-cache-dir --upgrade numpy pandas && \
    pip install --no-cache-dir \
  torch==${TORCH_VERSION} \
  torchvision==${TORCHVISION_VERSION} \
  torchaudio==${TORCHAUDIO_VERSION} \
  ultralytics==${ULTRALYTICS_VERSION} \
  opencv-python-headless==${OPENCV_HEADLESS_VERSION} \
  pyyaml==${PYYAML_VERSION} \
  pillow==${PILLOW_VERSION} \
  matplotlib==${MATPLOTLIB_VERSION} \
  scipy==${SCIPY_VERSION} \
  onnx==${ONNX_VERSION} \
  onnxruntime==${ONNXRUNTIME_VERSION} && \
  if [ "$INSTALL_WANDB" = "true" ]; then pip install --no-cache-dir wandb; fi
