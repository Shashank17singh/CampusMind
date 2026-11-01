# Build context: the repository root.
FROM python:3.11-slim AS builder
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.11-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 campusmind
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY backend/ .
RUN mkdir -p /data/uploads /data/huggingface && chown -R campusmind:campusmind /data /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    UPLOAD_DIR=/data/uploads
USER campusmind
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
