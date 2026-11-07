FROM ubuntu:24.04

# python3 runs the fake model provider (fake_llm/); curl is the environment
# plane's readiness probe. Nothing here talks to a real model.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY fake_llm /opt/fake_llm

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
