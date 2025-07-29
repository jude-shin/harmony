
FROM nvcr.io/nvidia/tensorflow:24.04-tf2-py3
WORKDIR /app

# Copy and install Python dependencies
COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

# CMD ["python3", "/harmony/src/manager.py"]

