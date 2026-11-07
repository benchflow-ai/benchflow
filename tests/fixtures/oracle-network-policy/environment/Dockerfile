# python:3.12-slim: the egress proxy that enforces the policy needs python3
# and openssl in the image.
FROM python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
