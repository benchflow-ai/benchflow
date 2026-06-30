# casinobench-base — the in-sandbox CasinoBench image.
#
# Extends the agent-seat image (examples/casino/agent_env, tagged
# `casino-agent-seat:latest`) so ONE shared sandbox carries both halves:
#   - the ACP agent backends (codex-acp / claude-agent-acp / deepagents) + the
#     `casino` seven-tool CLI  — inherited from casino-agent-seat, untouched;
#   - a WORKING `casino-service` (the env-0 HTTP mock-service on :9001).
#
# Why this image exists: casino-agent-seat installs env_0_casino with
# `pip install --no-deps`, so `casino-service` is a broken stub — it crashes on
#   from env_0_casino.app import create_app   (fastapi + the casinobench engine
# were never installed). Here we bake the casinobench engine + its `service`
# extra (fastapi / uvicorn / click / httpx) so `casino-service --help` exits 0
# and the service actually serves. env_0_casino itself is already installed in
# the base, so its console scripts (`casino`, `casino-service`) just light up.
#
# Build context = examples/casino/agent_env (same as casino-agent-seat). Assemble
# it first (see benchmarks/casinobench/README.md): the deepagents shim, the
# env_0_casino package as `casino-pkg/`, AND the casinobench engine checkout as
# `casinobench-engine/`. Then:
#   docker build -t casino-agent-seat:latest examples/casino/agent_env \
#     -f examples/casino/agent_env/Dockerfile
#   docker build -t casinobench-base:latest examples/casino/agent_env \
#     -f benchmarks/casinobench/docker/casinobench-base.Dockerfile
FROM casino-agent-seat:latest

# The casinobench engine (pure-Python, deterministic kernel — zero runtime deps)
# plus its `service` extra. `[service]` == fastapi + uvicorn[standard] + click +
# httpx, exactly the deps env_0_casino's server.py imports. The engine is
# proprietary (github.com/benchflow-ai/casinobench), so it is vendored into the
# build context rather than pulled from a public index.
COPY casinobench-engine /opt/casinobench-engine
RUN pip install --no-cache-dir "/opt/casinobench-engine[service]" && \
    python -c "import fastapi, uvicorn, casinobench.catalog" && \
    casino-service --help >/dev/null && \
    casino --help >/dev/null && \
    echo "casino-service + casino cli ok"

WORKDIR /app
