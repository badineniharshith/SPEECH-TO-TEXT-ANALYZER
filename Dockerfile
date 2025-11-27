# Use a slim Python image
FROM python:3.11-slim

# Install system deps (ffmpeg + build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dep file and install first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code and static files
COPY . /app

# Expose default port (Render provides $PORT at runtime)
ENV PORT=10000
EXPOSE 10000

# Use Gunicorn with Uvicorn workers (FastAPI)
# Use 1 worker to avoid multiple copies of Whisper model in memory
CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 1 --timeout 120"]
