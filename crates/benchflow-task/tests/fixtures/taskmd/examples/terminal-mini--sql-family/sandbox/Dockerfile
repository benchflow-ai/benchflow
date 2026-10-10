FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /data /work
WORKDIR /work
