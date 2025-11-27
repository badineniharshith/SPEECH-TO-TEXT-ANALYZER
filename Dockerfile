# Use a slim Python base
FROM python:3.11-slim

# Install system deps (ffmpeg + build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install deps
COPY requirements.txt /app/requirements.txt

# Upgrade pip then install
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and static files
COPY . /app

# Expose port (Render will provide $PORT)
ENV PORT=10000
EXPOSE 10000

# Use Gunicorn with Uvicorn workers and 1 worker (so the Whisper model is loaded only once)
CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 1 --timeout 120"]
