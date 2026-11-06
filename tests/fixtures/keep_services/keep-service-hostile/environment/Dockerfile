FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
RUN pip install --no-cache-dir pytest==8.3.3
# uid 1000 is taken, as by ubuntu:24.04's own user, so the agent user gets
# 1001: the uid a Linux host's second user has, whose folder /logs/verifier
# is on Docker (K2-R1-B-1).
RUN useradd -M -u 1000 filler
COPY server.py /opt/svc/server.py
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
