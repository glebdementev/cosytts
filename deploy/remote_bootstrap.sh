#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE_PATH="api/docker-compose.streaming.yml"
COMPOSE_CMD=(docker compose -f "${COMPOSE_FILE_PATH}")

echo "Remote bootstrap starting..."
echo "Stopping existing containers (if any)..."
${COMPOSE_CMD[@]} down --remove-orphans 2>/dev/null || true
echo "Building images (no cache)..."
${COMPOSE_CMD[@]} build
echo "Starting containers..."
${COMPOSE_CMD[@]} up -d
sleep 2
echo "Container status:"
${COMPOSE_CMD[@]} ps -a
echo "Container logs (last 50 lines):"
${COMPOSE_CMD[@]} logs --tail=50 api || true

HEALTH_URL="http://localhost:8090/health"
HEALTH_RETRIES=20
HEALTH_DELAY=30

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required for health checks." >&2
  exit 1
fi

echo "Verifying health endpoint (warmup may take a few minutes): ${HEALTH_URL}"
for attempt in $(seq 1 "${HEALTH_RETRIES}"); do
  response="$(curl -fsS "${HEALTH_URL}" || true)"
  if [[ "${response}" == *'"status":"ok"'* ]]; then
    echo "Health OK: ${HEALTH_URL}"
    break
  fi
  if [[ "${attempt}" -eq "${HEALTH_RETRIES}" ]]; then
    echo "ERROR: health check failed after ${HEALTH_RETRIES} attempts." >&2
    ${COMPOSE_CMD[@]} ps
    ${COMPOSE_CMD[@]} logs --tail=200 api || true
    exit 1
  fi
  sleep "${HEALTH_DELAY}"
done

echo "OK"
echo "Health:  http://<host>:8090/health"
echo "TTS:     http://<host>:8090/synthesize/stream"
echo "Remote bootstrap complete."
