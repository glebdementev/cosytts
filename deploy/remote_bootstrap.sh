#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE_PATH="api/docker-compose.streaming.yml"
COMPOSE_CMD=(docker compose -f "${COMPOSE_FILE_PATH}")
MODEL_DIR="pretrained_models/Fun-CosyVoice3-0.5B"
MODEL_ID="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

# ── 1. Download model if missing ─────────────────────────────────────────────
download_model() {
  local required_files=(
    cosyvoice3.yaml llm.pt flow.pt hift.pt
    campplus.onnx speech_tokenizer_v3.onnx spk2info.pt
  )
  local need_download=false

  if [[ ! -d "${MODEL_DIR}" ]]; then
    need_download=true
  else
    for f in "${required_files[@]}"; do
      if [[ ! -f "${MODEL_DIR}/${f}" ]]; then
        echo "Missing ${MODEL_DIR}/${f}"
        need_download=true
        break
      fi
    done
    if [[ ! -f "${MODEL_DIR}/CosyVoice-BlankEN/model.safetensors" ]] \
       && [[ ! -f "${MODEL_DIR}/CosyVoice-BlankEN/pytorch_model.bin" ]]; then
      echo "Missing CosyVoice-BlankEN weights"
      need_download=true
    fi
  fi

  if [[ "${need_download}" == "true" ]]; then
    echo "Downloading model ${MODEL_ID} into ${MODEL_DIR} ..."
    # Ubuntu 24.04 uses externally-managed Python (PEP 668),
    # so install modelscope in a local venv instead of system Python.
    local venv_dir=".cache/modelscope-venv"
    if ! python3 -m venv "${venv_dir}" 2>/dev/null; then
      apt-get update
      apt-get install -y --no-install-recommends python3-venv
      python3 -m venv "${venv_dir}"
    fi
    "${venv_dir}/bin/python" -m pip install --quiet --upgrade pip
    "${venv_dir}/bin/python" -m pip install --quiet modelscope
    "${venv_dir}/bin/python" << 'PYEOF'
from modelscope import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                  local_dir="pretrained_models/Fun-CosyVoice3-0.5B")
PYEOF
    echo "Model download complete."
  else
    echo "Model files already present in ${MODEL_DIR}, skipping download."
  fi
}

download_model

# ── 2. Build & start containers ──────────────────────────────────────────────
echo "Remote bootstrap starting..."
echo "Stopping existing containers (if any)..."
${COMPOSE_CMD[@]} down --remove-orphans 2>/dev/null || true
echo "Building images..."
${COMPOSE_CMD[@]} build
echo "Starting containers..."
${COMPOSE_CMD[@]} up -d
sleep 2
echo "Container status:"
${COMPOSE_CMD[@]} ps -a
echo "Container logs (last 50 lines):"
${COMPOSE_CMD[@]} logs --tail=50 api || true

# ── 3. Health check ──────────────────────────────────────────────────────────
API_PORT="${TTS_API_PORT:-8093}"
HEALTH_URL="http://localhost:${API_PORT}/health"
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
echo "Health:  http://<host>:${API_PORT}/health"
echo "TTS:     http://<host>:${API_PORT}/synthesize/stream"
echo "Remote bootstrap complete."
