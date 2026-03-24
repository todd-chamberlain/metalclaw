#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/model.gguf}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:--1}"
AGENT_TYPE="${AGENT_TYPE:-none}"
GPU_BACKEND="${GPU_BACKEND:-vulkan}"
INFERENCE_URL="${INFERENCE_URL:-}"
RPC_HOST="${RPC_HOST:-}"

echo "=== Metalclaw Sandbox ==="
echo "GPU backend: ${GPU_BACKEND}"
echo "Agent: ${AGENT_TYPE}"

# ── Metal mode: inference runs on the host ──────────────────────────
# Container only runs the agent/workspace, connecting to host API.
if [ "${GPU_BACKEND}" = "metal" ] && [ -n "${INFERENCE_URL}" ]; then
    echo "Inference: ${INFERENCE_URL} (host Metal GPU)"

    # Wait for host inference to be reachable
    echo "Waiting for host inference server..."
    for i in $(seq 1 60); do
        if curl -sf "${INFERENCE_URL}/health" >/dev/null 2>&1; then
            echo "Host inference server is ready"
            break
        fi
        sleep 2
    done

    if ! curl -sf "${INFERENCE_URL}/health" >/dev/null 2>&1; then
        echo "ERROR: Host inference server not reachable at ${INFERENCE_URL}"
        exit 1
    fi

    echo "API available at ${INFERENCE_URL}/v1"

    if [ "${AGENT_TYPE}" != "none" ]; then
        echo "Launching agent..."
        export PORT  # agent-wrapper uses PORT for local reference
        exec /usr/local/bin/agent-wrapper.sh
    else
        echo "Running in sandbox mode (inference on host)"
        echo "Use 'metalclaw connect' for interactive shell"
        # Keep container running
        exec sleep infinity
    fi
fi

# ── Vulkan / RPC / CPU mode: inference runs in-container ────────────
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Context: ${CTX_SIZE}"
echo "GPU layers: ${GPU_LAYERS}"

# Check GPU availability
if [ "${GPU_BACKEND}" = "vulkan" ]; then
    if command -v vulkaninfo &>/dev/null; then
        echo "Vulkan: $(vulkaninfo --summary 2>/dev/null | grep 'driverName' | head -1 || echo 'checking...')"
    else
        echo "Warning: vulkaninfo not found, GPU status unknown"
    fi
fi

# Build llama-server command
LLAMA_ARGS=(
    --model "${MODEL_PATH}"
    --port "${PORT}"
    --host 0.0.0.0
    --ctx-size "${CTX_SIZE}"
    --n-gpu-layers "${GPU_LAYERS}"
    --log-disable
)

# RPC mode: offload GPU work to host Metal via rpc-server
if [ -n "${RPC_HOST}" ]; then
    echo "RPC: offloading GPU to ${RPC_HOST}"
    LLAMA_ARGS+=(--rpc "${RPC_HOST}")
fi

# CPU mode: disable GPU layers
if [ "${GPU_BACKEND}" = "cpu" ]; then
    echo "Running in CPU-only mode"
    LLAMA_ARGS=(
        --model "${MODEL_PATH}"
        --port "${PORT}"
        --host 0.0.0.0
        --ctx-size "${CTX_SIZE}"
        --n-gpu-layers 0
        --log-disable
    )
fi

# Start llama-server
echo "Starting inference server..."
llama-server "${LLAMA_ARGS[@]}" &

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
