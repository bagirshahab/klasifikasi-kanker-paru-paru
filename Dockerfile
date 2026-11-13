# Backend LungVision AI untuk Hugging Face Spaces (Docker) / hosting container lain.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# PyTorch versi CPU (lebih kecil dari paket default)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r backend/requirements.txt

COPY backend backend
COPY frontend frontend

# HF Spaces menjalankan container sebagai user non-root; folder harus bisa ditulis
RUN mkdir -p backend/data backend/weights && chmod -R 777 backend/data backend/weights

WORKDIR /app/backend
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
