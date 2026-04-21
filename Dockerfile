FROM python:3.11-slim

LABEL maintainer="Language ID System"
LABEL description="Indic Language Identification & Script Analysis — AI4Bharat/Pralekha"

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p \
    artifacts/models \
    artifacts/eval \
    artifacts/analysis \
    cache/datasets \
    data/splits \
    logs

# Copy pre-trained models (if available)
# COPY artifacts/models/ ./artifacts/models/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 7860

# Default: API server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
