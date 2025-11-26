# Use official Python slim image
FROM python:3.11-slim

# Install ffmpeg and system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy project files
COPY . /app

# Expose the port
ENV PORT=8080
EXPOSE 8080

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
