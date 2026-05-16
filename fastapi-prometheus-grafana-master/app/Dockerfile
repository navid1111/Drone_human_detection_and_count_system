FROM python:3.11-slim

WORKDIR /app

# ----------------------------
# System stability tools
# ----------------------------
RUN apt-get update && apt-get install -y \
    curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Upgrade pip (IMPORTANT)
# ----------------------------
RUN pip install --upgrade pip

# ----------------------------
# Copy only requirements first (better caching)
# ----------------------------
COPY requirements.txt .

# ----------------------------
# Stable pip install for ML packages
# ----------------------------
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy rest of project
# ----------------------------
COPY . .

# ----------------------------
# Run server properly
# ----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]