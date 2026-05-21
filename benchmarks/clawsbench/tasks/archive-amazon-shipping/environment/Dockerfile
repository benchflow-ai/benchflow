# archive-amazon-shipping — per-task ClawsBench image.
#
# Builds FROM the smolclaws shared base. That base image is dependency-only:
# it bundles every third-party library claw-gmail needs (fastapi, sqlalchemy,
# faker, click, …) plus the `claw-gmail` console-script stub on PATH, but the
# smolclaws build (docker/generate_dockerfile.py) deliberately deletes the
# `claw_*` packages from site-packages. So `claw_gmail` is not importable in
# the base image — each per-task image must install the claw-* package it
# needs from source. This task only needs Gmail, so it vendors and installs
# `claw-gmail`.
#
# claw-gmail is vendored under environment/claw-gmail/ because the smolclaws
# repo (github.com/benchflow-ai/smolclaws) is private and cannot be reached by
# `pip install git+https://…` during an unauthenticated build.
#
# The image has NO service-starting entrypoint by design: the BenchFlow
# Environment plane (ManifestEnvironment, driven by
# benchmarks/clawsbench/environment.toml) probes `command -v claw-gmail`
# and starts the service itself inside the rollout sandbox.
FROM kywch/smolclaws-base:latest

# Install the vendored claw-gmail package into the image's system Python so
# `claw_gmail` is importable by /usr/local/bin/python (the interpreter behind
# the /usr/local/bin/claw-gmail console script). The base image already has
# every runtime dependency, so --no-build-isolation keeps the build offline
# and fast; a regular (non --no-deps) install is harmless since the deps are
# already satisfied.
COPY claw-gmail /tmp/claw-gmail
RUN uv pip install --system --no-build-isolation /tmp/claw-gmail && \
    rm -rf /tmp/claw-gmail && \
    uv cache clean && \
    python -c "import claw_gmail; print('claw_gmail import OK')"

# Seed deterministic Gmail data. The "default" scenario ships inside the
# claw-gmail package (claw_gmail/seed/content.py) — no smolclaws source tree
# is needed. seed=42 makes the message set reproducible across builds; that
# set always contains exactly one email from shipment-tracking@amazon.com.
RUN claw-gmail --db /data/gmail.db seed --scenario default --seed 42 && \
    test -f /data/gmail.db

# Log directories the BenchFlow verifier/agent expect.
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts

# The sandbox runs the agent as the non-root "agent" user; let it write logs.
RUN chown -R agent:agent /logs/agent /logs/artifacts

WORKDIR /app
