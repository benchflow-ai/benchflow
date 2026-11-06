FROM ubuntu:24.04
RUN apt-get update -qq && apt-get install -y -qq python3 python3-numpy && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY data/ /app/data/
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
