#!/bin/bash
# Deploy ep2-crypto to Hetzner VPS
#
# Usage:
#   ./scripts/deploy.sh              # Deploy (uses cached layers)
#   ./scripts/deploy.sh --build      # Force rebuild
#   ./scripts/deploy.sh --logs       # Show recent logs after deploy
#
# Prerequisites:
#   - SSH access configured in ~/.ssh/config as "hetzner"
#   - DOPPLER_TOKEN set in docker/.env on the server
#   - Docker + Docker Compose installed on server

set -euo pipefail

SERVER="${EP2_DEPLOY_HOST:-hetzner}"
DEPLOY_DIR="${EP2_DEPLOY_DIR:-/opt/ep2-crypto}"
HEALTH_URL="http://localhost:8000/health"
HEALTH_TIMEOUT=60
HEALTH_INTERVAL=5

BUILD_FLAG=""
SHOW_LOGS=false

for arg in "$@"; do
    case $arg in
        --build) BUILD_FLAG="--build" ;;
        --logs)  SHOW_LOGS=true ;;
        *)       echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "==> Syncing code to ${SERVER}:${DEPLOY_DIR}"
rsync -az --delete \
    --exclude .venv \
    --exclude __pycache__ \
    --exclude .git \
    --exclude data \
    --exclude research \
    --exclude '*.db' \
    --exclude .mypy_cache \
    --exclude .pytest_cache \
    --exclude .ruff_cache \
    --exclude htmlcov \
    . "${SERVER}:${DEPLOY_DIR}/"

echo "==> Starting services"
ssh "${SERVER}" "cd ${DEPLOY_DIR} && docker compose -f docker/docker-compose.yml up -d ${BUILD_FLAG}"

echo "==> Waiting for health check (timeout: ${HEALTH_TIMEOUT}s)"
elapsed=0
while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
    if ssh "${SERVER}" "curl -sf ${HEALTH_URL} > /dev/null 2>&1"; then
        echo "==> Deploy successful — health check passed (${elapsed}s)"
        if [ "$SHOW_LOGS" = true ]; then
            ssh "${SERVER}" "cd ${DEPLOY_DIR} && docker compose -f docker/docker-compose.yml logs --tail 30"
        fi
        exit 0
    fi
    sleep $HEALTH_INTERVAL
    elapsed=$((elapsed + HEALTH_INTERVAL))
done

echo "==> DEPLOY FAILED — health check did not pass within ${HEALTH_TIMEOUT}s"
echo "==> Recent logs:"
ssh "${SERVER}" "cd ${DEPLOY_DIR} && docker compose -f docker/docker-compose.yml logs --tail 50"
exit 1
