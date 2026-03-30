#!/usr/bin/env bash
set -euo pipefail

AGENT_TYPE="${AGENT_TYPE:-none}"
AGENT_COMMAND="${AGENT_COMMAND:-}"
PORT="${PORT:-8080}"
INFERENCE_URL="${INFERENCE_URL:-http://localhost:${PORT}}"

case "${AGENT_TYPE}" in
    openclaw)
        echo "Starting OpenClaw agent..."
        echo "Inference: ${INFERENCE_URL}/v1"

        # Configure OpenClaw to use local inference
        export OPENAI_API_BASE="${INFERENCE_URL}/v1"
        export OPENAI_API_KEY="metalclaw-local"
        export OPENCLAW_MODEL="local"

        # Try the CLI first, fall back to repo
        if command -v openclaw &>/dev/null; then
            exec openclaw start \
                --provider openai \
                --api-base "${INFERENCE_URL}/v1" \
                --api-key "metalclaw-local"
        elif [ -d /opt/openclaw ]; then
            cd /opt/openclaw
            exec node src/index.js
        else
            echo "ERROR: OpenClaw not found"
            exec /bin/bash
        fi
        ;;
    claude-code)
        echo "Starting Claude Code agent..."
        echo "Connecting to local inference at ${INFERENCE_URL}/v1"
        exec claude \
            --model "openai/local" \
            --api-base "${INFERENCE_URL}/v1"
        ;;
    mattermost)
        echo "Starting Mattermost bot bridge..."
        echo "Inference: ${INFERENCE_URL}/v1"
        exec python3 /usr/local/bin/mattermost-bot.py
        ;;
    custom)
        if [ -z "${AGENT_COMMAND}" ]; then
            echo "ERROR: AGENT_COMMAND not set for custom agent type"
            echo "Falling back to interactive shell"
            exec /bin/bash
        fi
        # Security: validate command doesn't contain shell metacharacters.
        # The Python side (container.py) already validates this, but defense-in-depth.
        if echo "${AGENT_COMMAND}" | grep -qE '[\$\`\;\|\&\<\>]|\$\('; then
            echo "ERROR: AGENT_COMMAND contains disallowed shell metacharacters"
            echo "Use a simple command without shell operators."
            exec /bin/bash
        fi
        echo "Starting custom agent: ${AGENT_COMMAND}"
        # Disable globbing to prevent * and ? expansion, then word-split the
        # validated command into argv for exec (no bash -c, no shell eval).
        set -f
        # shellcheck disable=SC2086
        exec ${AGENT_COMMAND}
        ;;
    *)
        echo "No agent configured, starting interactive shell"
        exec /bin/bash
        ;;
esac
