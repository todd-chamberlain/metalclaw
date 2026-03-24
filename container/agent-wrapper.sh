#!/usr/bin/env bash
set -euo pipefail

AGENT_TYPE="${AGENT_TYPE:-none}"
AGENT_COMMAND="${AGENT_COMMAND:-}"
PORT="${PORT:-8080}"

case "${AGENT_TYPE}" in
    claude-code)
        echo "Starting Claude Code agent..."
        echo "Connecting to local inference at http://localhost:${PORT}/v1"
        exec claude \
            --model "openai/local" \
            --api-base "http://localhost:${PORT}/v1"
        ;;
    custom)
        if [ -z "${AGENT_COMMAND}" ]; then
            echo "ERROR: AGENT_COMMAND not set for custom agent type"
            echo "Falling back to interactive shell"
            exec /bin/bash
        fi
        echo "Starting custom agent: ${AGENT_COMMAND}"
        exec bash -c "${AGENT_COMMAND}"
        ;;
    *)
        echo "No agent configured, starting interactive shell"
        exec /bin/bash
        ;;
esac
