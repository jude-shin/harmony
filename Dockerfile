# syntax=docker/dockerfile:1.7
########################
# 1. Base image layer  #
########################
FROM python:3.11-slim AS base
WORKDIR /app

# Install OS deps only once
RUN apt-get update && \
			apt-get install -y --no-install-recommends build-essential && \
			rm -rf /var/lib/apt/lists/*

########################
# 2. Python deps layer #
########################
FROM base AS deps

# Leverage Docker cache: first copy only dependency files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

########################
# 3. Runtime layer     #
########################
FROM base AS runtime
ENV PYTHONUNBUFFERED=1\
    PYTHONPATH=/app/src\
    MODEL_DIR=/models\
    DATA_DIR=/data\
    PORT=8000

# Copy installed python libs from previous stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy only the code needed for inference
COPY src/api /app/src/api
COPY src/inference /app/src/inference
COPY src/utils /app/src/utils
COPY src/config /app/src/config

EXPOSE ${PORT}

# Use gunicorn for production.  Adjust workers based on CPU cores.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api.server:app", "--bind", "0.0.0.0:8000", "--workers", "2"]

