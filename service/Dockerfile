FROM python:3.11

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV TF_CPP_MIN_LOG_LEVEL=2
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=2", "--threads=4", "src.api.app:app"]

