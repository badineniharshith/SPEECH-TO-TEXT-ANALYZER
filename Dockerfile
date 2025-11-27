# Use a slim Python image
FROM python:3.11-slim

# Reduce thread/blas parallelism (helps on low-RAM instances)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

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

# Optional: pre-download tiny model at build time (uncomment to bake model into image)
# RUN python -c "import whisper; whisper.load_model('tiny')"

EXPOSE 10000

# Use one worker (prevents multiple heavy model loads); bind to PORT env variable provided by Render
CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --workers 1 --timeout 120"]
