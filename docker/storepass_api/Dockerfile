FROM python:3.11-slim AS base
WORKDIR /harmony

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements requirements
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/storepass_api.txt

COPY src/storepass_api src/storepass_api
COPY src/utils src/utils

CMD ["fastapi", "run", "src/storepass_api/main.py", "--port", "--proxy-headers", "80"]

# EXPOSE ${SERVING_API_PORT}
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.serving_api.server:app", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "120"]

