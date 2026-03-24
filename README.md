# Metalclaw

Sandboxed GPU-accelerated AI agent runtime for Apple Silicon.

Metalclaw spins up a rootless Podman container with GPU passthrough (via libkrun + Venus/Vulkan), runs local LLM inference with llama.cpp, and enforces network policies — all on your Mac with zero cloud dependencies.

Inspired by NVIDIA's NemoClaw, rebuilt from scratch for Apple Metal.

## Requirements

- **Apple Silicon** Mac (M1/M2/M3/M4, any tier)
- **macOS 14+**
- **Podman 5.0+** — `brew install podman`
- **krunkit** — `brew install krunkit`
- **Python 3.11+**
- **20 GB+ free disk** (plus model storage)

For 70B+ models: M-series Ultra or Pro with 64 GB+ unified memory recommended.

## Install

```bash
git clone https://github.com/todd-chamberlain/metalclaw.git
cd metalclaw
pip install -e .
```

## Quick Start

```bash
# One-time setup: preflight checks, GPU detection, model download, container build
metalclaw onboard

# Start the sandbox
metalclaw run

# In another terminal
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model.gguf","messages":[{"role":"user","content":"hello"}]}'

# Shell into the sandbox
metalclaw connect

# Check status
metalclaw status

# Shut down
metalclaw stop
```

## Models

Metalclaw ships with a curated registry of GGUF models. All models use Q4_K_M quantization.

| Key | Model | Size | Min RAM | Notes |
|-----|-------|------|---------|-------|
| `qwen2.5-7b` | Qwen 2.5 7B Instruct | 4.7 GB | 8 GB | Fast, good for testing |
| `qwen2.5-72b` | Qwen 2.5 72B Instruct | 42 GB | 64 GB | Strong general-purpose |
| `qwen3-coder-next` | Qwen 3 Coder Next | 48.5 GB | 64 GB | Coding model, thinking modes |
| `llama-3.3-70b` | Llama 3.3 70B Instruct | 40 GB | 64 GB | Large context (131K) |
| `deepseek-r1-70b` | DeepSeek R1 70B | 40 GB | 64 GB | Reasoning and code |

```bash
# List models and download status
metalclaw model list

# Download a model
metalclaw model pull qwen3-coder-next

# Run with a specific model
metalclaw run --model qwen3-coder-next
```

Downloads resume automatically if interrupted.

## Network Policies

Metalclaw enforces network policies on the sandbox container. The default policy is deny-all with localhost inference access.

```bash
# List available policies
metalclaw policy list

# Show policy details
metalclaw policy show github

# Run with additional policy presets
metalclaw run --presets github,pypi
```

**Built-in presets:**

| Preset | Allows |
|--------|--------|
| `default` | Deny-all, localhost only |
| `github` | github.com, api.github.com, *.githubusercontent.com |
| `pypi` | pypi.org, files.pythonhosted.org |
| `npm` | registry.npmjs.org |
| `anthropic` | api.anthropic.com (for Claude Code agent) |

## Agent Mode

Metalclaw can run an AI agent inside the sandbox that connects to the local inference server.

```bash
# Run with Claude Code agent
metalclaw run --agent claude-code --presets anthropic

# Run with a custom command
metalclaw run --agent custom
```

Agent types:
- **`none`** (default) — inference server only, connect from outside
- **`claude-code`** — runs Claude Code CLI inside the sandbox, pointed at the local llama-server
- **`custom`** — runs the command specified in config `agent.command`

## CLI Reference

```
metalclaw onboard          One-time setup wizard
metalclaw run              Start the sandbox with inference
metalclaw status           Show machine, container, and inference status
metalclaw stop             Stop the container (--machine to also stop the VM)
metalclaw connect          Shell into the running sandbox
metalclaw logs             Show container logs (-f to follow)
metalclaw model list       List available models
metalclaw model pull KEY   Download a model
metalclaw policy list      List policy presets
metalclaw policy show NAME Show policy details
```

## Configuration

Config lives at `~/.metalclaw/config.yaml`. Created automatically on first run.

```yaml
version: 1
sandbox:
  name: metalclaw-sandbox
  memory_limit: 64g
  cpus: 8
gpu:
  backend: vulkan
  layers: -1              # -1 = all layers on GPU
inference:
  model: qwen2.5-7b       # registry key or path to .gguf
  port: 8080
  context_size: 8192
agent:
  type: none               # none | claude-code | custom
  command: ""
policy:
  base: default
  presets: []
machine:
  provider: libkrun
  cpus: 8
  memory: 61440            # MiB (krunkit max ~60 GB)
  disk: 100                # GB
```

Models are cached at `~/.metalclaw/models/`.

## Architecture

```
┌─────────────────────────────────────────────┐
│  macOS Host                                 │
│                                             │
│  metalclaw CLI (Python)                     │
│       │                                     │
│       ▼                                     │
│  Podman + libkrun (micro-VM)               │
│  ┌─────────────────────────────────────┐   │
│  │  Fedora 40 Container               │   │
│  │                                     │   │
│  │  llama-server (ggml-vulkan)        │   │
│  │    ├── /models/model.gguf (ro)     │   │
│  │    └── :8080 OpenAI-compatible API │   │
│  │                                     │   │
│  │  Venus/Virtio-GPU ──► Metal GPU    │   │
│  │                                     │   │
│  │  Network: policy-enforced          │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  127.0.0.1:8080/v1 ◄── API access         │
└─────────────────────────────────────────────┘
```

**Key components:**
- **Podman + libkrun** — rootless micro-VM with GPU passthrough, stronger isolation than containers alone
- **Fedora 40** — only distro with patched mesa-vulkan-drivers for Venus in containers
- **llama.cpp** — compiled with `-DGGML_VULKAN=ON`, serves an OpenAI-compatible API
- **Network policies** — NemoClaw-compatible YAML schema, enforced via Podman network modes

## API

The inference server exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

```bash
# Health check
curl http://127.0.0.1:8080/health

# List models
curl http://127.0.0.1:8080/v1/models

# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 100
  }'

# Streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

Works with any OpenAI-compatible client — just set the base URL to `http://127.0.0.1:8080/v1`.

## Project Structure

```
metalclaw/
├── pyproject.toml
├── src/metalclaw/
│   ├── cli.py          # Click CLI
│   ├── config.py       # ~/.metalclaw/config.yaml management
│   ├── preflight.py    # System requirement checks
│   ├── gpu.py          # Apple Silicon GPU detection
│   ├── machine.py      # Podman machine lifecycle (libkrun)
│   ├── container.py    # Container image build and lifecycle
│   ├── models.py       # GGUF model registry and download
│   ├── inference.py    # llama-server health checks
│   ├── policy.py       # Network policy parsing and enforcement
│   └── agent.py        # Agent runtime config
├── container/
│   ├── Containerfile    # Fedora 40 + Vulkan + llama.cpp
│   ├── metalclaw-start.sh
│   └── agent-wrapper.sh
├── policies/
│   ├── default.yaml
│   └── presets/
│       ├── github.yaml
│       ├── pypi.yaml
│       ├── npm.yaml
│       └── anthropic.yaml
└── tests/
```

## License

MIT
