# ep2-crypto production image
# Multi-stage build: python:3.12-slim + uv + Doppler CLI

FROM python:3.12-slim AS base

# System dependencies for TimescaleDB driver (psycopg2) and C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Doppler CLI for secrets injection
RUN curl -sLf --retry 3 --tlsv1.2 --proto "=https" \
    "https://cli.doppler.com/install.sh" | sh

WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen = exact lockfile versions)
RUN uv sync --frozen --extra all --extra polymarket --no-dev --no-install-project

# Copy application code
COPY src/ src/
COPY scripts/ scripts/

# Install the project itself
RUN uv sync --frozen --extra all --extra polymarket --no-dev

# Non-root user for security
RUN useradd --create-home --shell /bin/bash ep2
USER ep2

EXPOSE 8000

# Health check against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default: run live loop with API server via Doppler secrets injection
CMD ["doppler", "run", "--", "uv", "run", "python", "scripts/live.py", "--api", "--venue", "polymarket_binary"]
