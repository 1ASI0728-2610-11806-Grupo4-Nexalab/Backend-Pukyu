FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./simulation /code/simulation
COPY ./download_model.py /code/download_model.py

# Pre-descargar el modelo YOLO para acelerar el inicio en Render
RUN python download_model.py

# Iniciar el servidor
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
