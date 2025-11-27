# Use a slim Python image
FROM python:3.11-slim

# Install system deps (ffmpeg + build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install numpy first to avoid binary mismatch issues
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy code and static files
COPY . /app

# Optionally pre-download a small model at build time (uncomment to bake 'tiny' into image)
# WARNING: increases image size and build time
# RUN python -c "import whisper; whisper.load_model('tiny')"

# Render exposes $PORT at runtime; set a default for local testing
ENV PORT=10000
EXPOSE 10000

# Use Gunicorn with Uvicorn worker for FastAPI. Use 1 worker so the model is loaded only once.
CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 1 --timeout 120"]
