FROM python:3.12-slim

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# The simulator plugin is copied into this build context by bench/sync_plugin.sh.
COPY metal-sim /opt/metal-sim
RUN pip install --no-cache-dir "inspect-robots==0.58.0" "inspect-robots-agent==0.26.0" /opt/metal-sim

WORKDIR /app
COPY policy.py /app/policy.py

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
