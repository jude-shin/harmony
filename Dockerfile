# CUDA + Python 3.11
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/models \
    DATA_DIR=/data \
    PORT=8000

# OS deps + Python + fresh pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3.11-dev \
        build-essential libgl1 libglib2.0-0 curl ca-certificates && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Dependencies stage
FROM base AS deps
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM base AS runtime

# Bring in everything pip put under /usr/local (binaries + wheels)
COPY --from=deps /usr/local /usr/local

# Your source tree
COPY src/api /app/src/api
COPY src/inference /app/src/inference
COPY src/utils /app/src/utils
COPY src/helper /app/src/helper
COPY src/harmony_config /app/src/harmony_config

EXPOSE ${PORT}

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api.server:app", "--workers", "1", "--bind", "0.0.0.0:8000", "--timeout", "120"]

