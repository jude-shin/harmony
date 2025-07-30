FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
WORKDIR /harmony

COPY requirements/base.txt /harmony/requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

# temporary fix for upstream numpy issue?
RUN pip install "numpy<2"

COPY src/cnn src/cnn
COPY src/manager.py src/manager.py
COPY src/data src/data
COPY src/training src/training
COPY src/utils src/utils

CMD ["python3", "src/manager.py"]
