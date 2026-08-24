# ─────────────────────────────────────────────
# Stage 1: Builder — install deps & download models
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Pre-download model weights so they're baked into the image
COPY scripts/download_models.py ./scripts/
RUN python scripts/download_models.py --cache-dir /build/models || true

# ─────────────────────────────────────────────
# Stage 2: Runtime — slim production image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded models
COPY --from=builder /build/models /app/models

# Copy application source
COPY src/ /app/src/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
