# Dockerfile (place at repo root)
FROM python:3.11-slim

# Install ffmpeg and build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU wheels from PyTorch index (use matching versions)
# Note: using --extra-index-url ensures pip can find +cpu wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2+cpu" "torchaudio==2.2.2+cpu" "torchvision==0.17.0+cpu"

# Install the remaining Python requirements
RUN pip install -r /app/requirements.txt

# Copy app code
COPY . /app

# Expose port and env
ENV PORT=8080
EXPOSE 8080

# Use the model env var if you want to change the model from Render later
ENV WHISPER_MODEL=base

# Start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
