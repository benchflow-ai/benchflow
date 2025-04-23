# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    file \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/jfk_text && \
    mkdir -p /app/backend/cache && \
    chmod -R 777 /app/data && \
    chmod -R 777 /app/backend/cache

# Copy data files first
COPY data/jfk_text/ /app/data/jfk_text/

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 