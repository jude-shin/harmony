# Start from a standard Python image
FROM python:3.11-slim AS base
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements/base.txt .
COPY requirements/api.txt .
RUN pip install --no-cache-dir -r base.txt -r api.txt

# Copy your app code
COPY src/ /app/src

# Expose port and start app
EXPOSE ${API_PORT}
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api.server:app", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "120"]

