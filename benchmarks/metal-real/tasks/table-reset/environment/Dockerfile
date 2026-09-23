FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git bash && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY robot.py /app/robot.py
