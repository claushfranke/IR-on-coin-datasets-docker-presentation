FROM python:3.11-slim

# Systemabhängigkeiten für OpenCV (libgl1-mesa-glx wurde durch libgl1 ersetzt ab Debian 12+)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend-Code kopieren
COPY app.py .
COPY models.py .
COPY utils.py .
COPY download_models.py .

# Verzeichnisse für Modelle und temporäre Dateien
RUN mkdir -p /app/models /tmp/coin_analysis

# Umgebungsvariablen
ENV MODELS_DIR=/app/models
ENV IMAGES_DIR=/app/images
ENV PYTHONUNBUFFERED=1

# Port exponieren
EXPOSE 8000

# Health-Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/api/health'); r.raise_for_status()"

# Server starten
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
