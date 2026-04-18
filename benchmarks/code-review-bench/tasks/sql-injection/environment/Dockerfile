FROM python:3.12-slim

RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY user_service.py /app/user_service.py

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
