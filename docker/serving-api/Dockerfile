FROM python:3.11-slim AS base
WORKDIR /harmony

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements requirements
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/api.txt -r requirements/tf.txt

COPY src/api src/api
COPY src/utils src/utils
COPY src/processing src/processing

EXPOSE ${SERVING_API_PORT}
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api.server:app", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "120"]

