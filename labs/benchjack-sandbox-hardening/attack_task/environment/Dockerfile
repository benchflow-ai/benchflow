FROM python:3.12-slim

RUN pip install --no-cache-dir pytest==8.3.3

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app/exploit

COPY conftest_exploit.py /app/exploit/conftest.py

WORKDIR /app
