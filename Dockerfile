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

# Iniciar el servidor leyendo el puerto dinámico asignado por la plataforma (Render usa 10000, local usa 8000)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
