# Agent (`main`) container of an embodied task. It holds only the `robo` client (standard-library Python) plus numpy.
# No simulator, task source, episode record or verifier is present here: the robot is reachable only through the
# episode socket that the trusted `simulator` service shares at /rpc/episode.sock.
FROM python:3.12-slim-bookworm
# curl/xz/ca-certificates let BenchFlow install JS agent harnesses (Claude Code, Codex) without apt at run time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir numpy==2.5.3
COPY robo /usr/local/bin/robo
RUN chmod 755 /usr/local/bin/robo && mkdir -p /app /logs/agent /logs/verifier /logs/artifacts
ENV ROBO_SOCKET=/rpc/episode.sock ROBOUSE_SOCKET=/rpc/episode.sock PYTHONUNBUFFERED=1
WORKDIR /app
