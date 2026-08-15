FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY services ./services

RUN uv sync --frozen --no-dev --extra azure-service \
    && groupadd --system benchflow \
    && useradd --system --gid benchflow --no-create-home benchflow

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER benchflow

EXPOSE 8000

CMD ["uvicorn", "services.trajectory_upload.broker_app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
