#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/model.gguf}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:--1}"
AGENT_TYPE="${AGENT_TYPE:-none}"

echo "=== Metaclaw Sandbox ==="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Context: ${CTX_SIZE}"
echo "GPU layers: ${GPU_LAYERS}"
echo "Agent: ${AGENT_TYPE}"

# Verify Vulkan availability
if command -v vulkaninfo &>/dev/null; then
    echo "Vulkan: $(vulkaninfo --summary 2>/dev/null | grep 'driverName' | head -1 || echo 'checking...')"
else
    echo "Warning: vulkaninfo not found, GPU status unknown"
fi

# Start llama-server
echo "Starting inference server..."
llama-server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --ctx-size "${CTX_SIZE}" \
    --n-gpu-layers "${GPU_LAYERS}" \
    --log-disable \
    &

SERVER_PID=$!

# Wait for health check
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "Server is ready on port ${PORT}"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 2
done

# Verify server responded
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: Server did not become healthy in time"
    exit 1
fi

echo "API available at http://localhost:${PORT}/v1"

# Launch agent or keep server in foreground
if [ "${AGENT_TYPE}" != "none" ]; then
    echo "Launching agent..."
    exec /usr/local/bin/agent-wrapper.sh
else
    echo "Running in inference-only mode (no agent)"
    wait "${SERVER_PID}"
fi
