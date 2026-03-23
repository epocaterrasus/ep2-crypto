# ep2-crypto production image
# Multi-stage build: python:3.12-slim + uv + Doppler CLI

FROM python:3.12-slim AS base

# System dependencies for TimescaleDB driver (psycopg2) and C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Doppler CLI for secrets injection
RUN apt-get update && apt-get install -y --no-install-recommends apt-transport-https \
    && curl -sLf --retry 3 --tlsv1.2 --proto "=https" \
    'https://packages.doppler.com/public/cli/gpg.DE2A7741A397C129.key' \
    | gpg --dearmor -o /usr/share/keyrings/doppler-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/doppler-archive-keyring.gpg] https://packages.doppler.com/public/cli/deb/debian any-version main" \
    | tee /etc/apt/sources.list.d/doppler-cli.list \
    && apt-get update && apt-get install -y doppler \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — created BEFORE deps so venv is owned by ep2
RUN useradd --create-home --shell /bin/bash ep2

WORKDIR /app
RUN chown ep2:ep2 /app

USER ep2

# Copy dependency files first for Docker layer caching
COPY --chown=ep2:ep2 pyproject.toml uv.lock ./

# Install dependencies (frozen = exact lockfile versions)
# --extra all resolves to ml + ingest + api + polymarket (see pyproject.toml [all])
RUN uv sync --frozen --extra all --no-dev --no-install-project

# Copy application code
COPY --chown=ep2:ep2 src/ src/
COPY --chown=ep2:ep2 scripts/ scripts/

# Install the project itself
RUN uv sync --frozen --extra all --no-dev

# Persist trained models across container restarts via a named volume
RUN mkdir -p /app/models

EXPOSE 8000

# Health check against the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default: run live loop with API server via Doppler secrets injection
CMD ["doppler", "run", "--", "uv", "run", "python", "scripts/live.py", "--api", "--venue", "polymarket_binary"]
