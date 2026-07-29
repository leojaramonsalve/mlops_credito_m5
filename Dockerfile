# Dockerfile - Pipeline MLOps Credito M5
# ========================================
# Imagen para el endpoint FastAPI de prediccion de pago a tiempo.
# Base: Python 3.10 slim (minimal y compatible con todas las libs ML).
#
# Build:   docker build -t mlops_credito_m5:latest .
# Run:     docker run -p 8000:8000 mlops_credito_m5:latest
# Test:    curl http://localhost:8000/health

FROM python:3.10-slim

# Metadata
LABEL maintainer="Leonardo Jaramonsalve <github.com/leojaramonsalve>"
LABEL project="mlops_credito_m5"
LABEL version="1.2.0"
LABEL description="API FastAPI de prediccion de pago a tiempo - Modulo 5 Henry"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

# Crear usuario no-root (buena practica de seguridad)
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR ${APP_HOME}

# Instalar dependencias de sistema necesarias para lightgbm (libgomp1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (cache de capas)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el codigo del proyecto
COPY mlops_pipeline ./mlops_pipeline
COPY models ./models
COPY data_processed ./data_processed

# Cambiar propietario al usuario no-root
RUN chown -R appuser:appuser ${APP_HOME}

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Healthcheck (Docker reporta si el servicio esta sano)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio: uvicorn levanta FastAPI
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
