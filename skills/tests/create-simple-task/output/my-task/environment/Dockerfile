FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
