# Retail Object Detection MLOps Assignment

## Overview
This repository shows an end-to-end retail object detection workflow with an emphasis on reproducibility, deployment readiness, and observability.

The core focus is on what was implemented and validated:
- DVC for data and artifact versioning
- ONNX export for efficient, framework-agnostic inference
- FastAPI inference service with prediction logging
- Prometheus + Grafana monitoring for model and service behavior

## Colab Notebook
- Google Colab: https://colab.research.google.com/drive/1kQgV7Y1WdnaFq2ovuexW22Cbaxhha4hN?usp=sharing

## Getting Started

### Prerequisites
- Docker and Docker Compose installed
- Python 3.9+ (for local development, optional)

### Quick Start

#### Step 1: Start the Main Infrastructure
Navigate to the project root and start the main Docker containers:

```bash
# From the assignment root directory
docker-compose up -d
```

This starts:
- **Airflow**: http://localhost:8080 (username: `airflow`, password: `airflow`)
- **Streamlit Dashboard**: http://localhost:8501
- **MySQL Database**: localhost:3306 (credentials in docker-compose.yml)

#### Step 2: Start the FastAPI + Prometheus + Grafana Stack
In a new terminal, navigate to the FastAPI directory and start its services:

```bash
# Navigate to the FastAPI app directory
cd fastapi-prometheus-grafana-master

# Start FastAPI, Prometheus, and Grafana
docker-compose up
```

This starts:
- **FastAPI API**: http://localhost:8000 (API docs at `/docs`)
- **Prometheus**: http://localhost:9090 (metrics database)
- **Grafana**: http://localhost:3000 (dashboards, username: `admin`, password: `admin`)

### Verify Everything is Running

Check all services are healthy:

```bash
# Option 1: Check Docker containers
docker-compose ps  # From assignment root
cd fastapi-prometheus-grafana-master && docker-compose ps  # From FastAPI directory

# Option 2: Test API health
curl http://localhost:8000/health

# Option 3: Check Prometheus
curl http://localhost:9090/-/healthy
```

### First Steps with the API

#### 1. Access API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### 2. Run Your First Detection
```bash
# Upload an image for detection
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/image.jpg"
```

#### 3. View Results
```bash
# List recent predictions
curl http://localhost:8000/predictions

# View specific prediction (replace 1 with actual ID)
curl http://localhost:8000/predictions/1
```

#### 4. Check Prometheus Metrics
```bash
# View all collected metrics
curl http://localhost:8000/metrics
```

#### 5. Monitor in Grafana
1. Visit http://localhost:3000
2. Login with `admin/admin`
3. Navigate to Dashboards to view real-time metrics
4. Key dashboards show:
   - Prediction throughput
   - Model inference latency
   - Confidence score distribution
   - Detection performance by class

### Stopping Services

```bash
# Stop FastAPI stack (from fastapi-prometheus-grafana-master directory)
docker-compose down

# Stop main infrastructure (from assignment root)
docker-compose down
```

## Final Model Snapshot
- Dataset classes: 76
- mAP50: 0.919
- Precision: 0.906
- Recall: 0.918
- mAP50-95: 0.68

These results improved significantly after fixing data split quality.

### Experiment Tracking
All training runs, hyperparameter comparisons, and metrics are tracked in Weights & Biases:

📊 **[View Full Experiment Report on W&B](https://api.wandb.ai/links/navidkamal-islamic-university-of-technology/2odl87uh)**

The report includes:
- Run-to-run comparisons across different configurations
- Loss curves and validation metrics over time
- Hyperparameter impact analysis
- Model performance evolution

## What Was Built

### 1. Data Versioning with DVC
DVC is used to track large data/model artifacts outside Git while keeping reproducible pointers in the repository.

Why this matters:
- Reproducibility: anyone can pull the exact dataset/model version
- Traceability: model quality can be tied to specific data snapshots
- Collaboration: large files are managed without bloating Git history

Typical workflow:
```bash
# Pull tracked data/artifacts
dvc pull

# (when adding/updating artifacts)
dvc add dataset/
git add dataset.dvc .gitignore
git commit -m "Track dataset with DVC"
```

### 2. Airflow Workflow (Diagram Only)
Airflow orchestrates training/evaluation flow. The README intentionally keeps only the high-level workflow view.

```mermaid
flowchart TD
    A[check_data_quality] --> B{decide_rebuild}
    B -->|needs rebuild| C[rebuild_splits]
    B -->|skip| D[skip_rebuild]
    C --> E[join]
    D --> E
    E --> F[check_gpu]
    F --> G[train_model]
    G --> H[evaluate_model]
    H --> I[run_share_of_shelf]
    I --> J[push_to_registry]
```

Airflow UI:
- URL: `http://localhost:8080`
- DAG: `branch_dag`

![Airflow DAG Graph](branch_dag-graph.png)

### 3. ONNX Export and Why It Was Used
The trained YOLO model was exported to ONNX (`best.onnx`) and used for serving.

Why ONNX was chosen:
- Smaller, lighter inference stack for deployment
- No PyTorch runtime needed in inference container
- Faster startup and simpler Docker images
- Better portability across runtimes/hardware

Produced artifact:
- `runs/train/pipeline_run/weights/best.onnx`

### 4. FastAPI Inference + Prediction Logging

A FastAPI backend accepts images, runs ONNX inference, and stores prediction metadata in MySQL.
![FastAPI app UI](screencapture-localhost-8501-2026-03-11-20_03_13.png)
Logged information includes:
- Timestamp
- Inference latency
- Bounding boxes
- Class IDs/names
- Confidence scores
- Optional ground-truth payload

Main endpoints:
- `POST /predict`
- `GET /predictions`
- `GET /predictions/{id}`
- `GET /health`
- `GET /metrics` (Prometheus scrape endpoint)

### 5. Observability with Prometheus + Grafana
Prometheus collects application/inference metrics and Grafana visualizes them.

System-level/API observability:
- Request rate
- Error rate
- P50/P90 request latency
- CPU and memory usage

Model-level observability added:
- Prediction throughput
- Detection throughput by class
- Confidence score behavior
- Inference latency distributions
- Detections per image
- Low-confidence detection trends

This enables monitoring both service health and model quality drift signals.

![Grafana Model Observability Dashboard](screencapture-localhost-3000-d-eX4mpl3-fastapi-dashboard-2026-03-10-11_54_44.png)

## Training Curves
The training and validation curves below summarize optimization behavior and final metric convergence.

![Training and Validation Curves](runs/train/pipeline_run/results.png)

## Inference and Monitoring Stack
`fastapi-prometheus-grafana-master/` contains:
- `app/`: FastAPI + ONNX Runtime + MySQL logging
- `streamlit/`: simple UI for uploads and results
- `prometheus/`: scrape config
- `grafana/`: dashboard provisioning

## Run (Monitoring Stack)
From `fastapi-prometheus-grafana-master/`:

```bash
docker compose up -d --build
```

Services:
- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Repository Structure (High Level)
```text
assignment/
|- dags/
|- src/
|- dataset/
|- runs/
|- fastapi-prometheus-grafana-master/
|- docker-compose.yml
|- README.md
```

## Notes
- Large artifacts are expected to be restored via DVC.
- ONNX inference is intentionally used for serving to keep deployment lean.
- Dashboard is oriented toward both API uptime and model behavior, not only infra metrics.
