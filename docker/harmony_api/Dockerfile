FROM python:3.11-slim AS base
WORKDIR /harmony

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements requirements
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/harmony_api.txt -r requirements/tf.txt

COPY src/harmony_api src/harmony_api
COPY src/utils src/utils
COPY src/processing src/processing

# TODO: turn this form shell form to exec form
# CMD ["fastapi", "run", "src/serving_api/main.py", "--proxy-headers", "--port", $SERVING_API_PORT]
CMD fastapi run src/harmony_api/main.py --proxy-headers --port $HARMONY_API_PORT
