# Use an official Python image
FROM python:3.11-slim

# Install ffmpeg and system tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for caching
COPY requirements.txt .

# Upgrade pip and install deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code & static files
COPY . /app

# Expose default Render port (Render will provide $PORT at runtime)
EXPOSE 10000

# Use PORT env var when starting (Render sets PORT)
ENV PORT=10000

# Start uvicorn; bind to 0.0.0.0 and $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
