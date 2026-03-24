# Metalclaw

Sandboxed GPU-accelerated AI agent runtime for Apple Silicon.

Metalclaw runs local LLM inference with native Metal GPU acceleration inside a sandboxed Podman container, with NemoClaw-compatible network policy enforcement -- all on your Mac with zero cloud dependencies.

Inspired by NVIDIA's NemoClaw, rebuilt from scratch for Apple Metal.

## Requirements

- **Apple Silicon** Mac (M1/M2/M3/M4, any tier)
- **macOS 14+**
- **Podman 5.0+** -- `brew install podman`
- **krunkit** -- `brew install krunkit`
- **cmake** -- `brew install cmake`
- **Python 3.11+**
- **20 GB+ free disk** (plus model storage)

For 70B+ models: M-series Ultra or Pro with 64 GB+ unified memory recommended.

## Install

```bash
brew install podman krunkit cmake pipx

git clone https://github.com/todd-chamberlain/metalclaw.git
cd metalclaw
pipx install -e . --python python3.12
```

This puts `metalclaw` on your PATH immediately. Verify with:

```bash
metalclaw --version
```

## Quick Start

```bash
# One-time setup: preflight checks, GPU detection, Metal server build, model download
metalclaw onboard

# Start the sandbox with Metal GPU acceleration
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

## GPU Backends

Metalclaw supports three GPU backends:

| Backend | Speed | Description |
|---------|-------|-------------|
| `metal` (default) | **100% native** | llama-server runs on host with Metal GPU. Container handles agent/workspace isolation. |
| `vulkan` | ~50-60% native | llama-server runs in container with Vulkan via Venus/virtio-gpu pipeline. |
| `cpu` | ~20% native | CPU-only mode, no GPU acceleration. |

### Metal Mode (Default)

In metal mode, metalclaw builds llama.cpp natively on macOS with the Metal backend (`~/.metalclaw/bin/llama-server`) and runs it on the host for full Apple GPU performance. The sandbox container connects to the host inference server for agent execution and workspace isolation.

**Performance on M3 Ultra (Qwen 2.5 7B):**
- Prompt: 539 tok/s (6.7x faster than CPU)
- Generation: 106 tok/s (3.6x faster than CPU)

```bash
# Override GPU backend
metalclaw run --gpu metal     # Default - native Metal
metalclaw run --gpu vulkan    # In-container Vulkan
metalclaw run --gpu cpu       # CPU only

# Rebuild the Metal server
metalclaw build
metalclaw build --force       # Force rebuild
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

Metalclaw uses a NemoClaw-compatible YAML policy schema with deny-all default networking, composable presets, and filesystem/process isolation.

```bash
# List available policies
metalclaw policy list

# Show policy details
metalclaw policy show github

# Run with additional policy presets
metalclaw run --presets github,pypi
```

**Built-in presets:**

| Preset | Allows | Binary Restrictions |
|--------|--------|-------------------|
| `default` | Deny-all, localhost only | -- |
| `github` | github.com, api.github.com, *.githubusercontent.com | git, gh, curl |
| `pypi` | pypi.org, files.pythonhosted.org | pip3, python3 |
| `npm` | registry.npmjs.org, *.npmjs.com | npm, npx |
| `anthropic` | api.anthropic.com, *.anthropic.com | claude |

### Policy Schema

Policies use a NemoClaw-compatible YAML format with three sections:

```yaml
version: 1
name: example
description: Example sandbox policy

# Filesystem isolation (podman --read-only + tmpfs)
filesystem_policy:
  read_only_root: true
  read_write: [/sandbox, /tmp]
  read_only: [/usr, /lib, /etc]

# Process isolation (podman --user)
process:
  run_as_user: sandbox
  run_as_group: sandbox

# Network isolation (podman pasta network mode)
network_policies:
  default_action: deny
  allow_localhost: true
  allowed_endpoints:
    - host: api.github.com
      ports: [443]
      protocol: tcp
      direction: outbound
      binaries: [/usr/bin/git]      # restrict which binaries can use this endpoint
      access: read-only             # read-only = GET/HEAD/OPTIONS only
      tls: terminate                # enable L7 HTTP method inspection
```

## Agent Mode

Metalclaw can run an AI agent inside the sandbox that connects to the local inference server.

```bash
# Run with Claude Code agent
metalclaw run --agent claude-code --presets anthropic

# Run with a custom command
metalclaw run --agent custom
```

Agent types:
- **`none`** (default) -- inference server only, connect from outside
- **`claude-code`** -- runs Claude Code CLI inside the sandbox, pointed at the local llama-server
- **`custom`** -- runs the command specified in config `agent.command`

## CLI Reference

```
metalclaw onboard          One-time setup wizard
metalclaw run              Start the sandbox with inference
metalclaw status           Show machine, container, and inference status
metalclaw stop             Stop the container (--machine to also stop the VM)
metalclaw connect          Shell into the running sandbox
metalclaw logs             Show container logs (-f to follow)
metalclaw build            Build/rebuild the host Metal inference server
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
  backend: metal           # metal | vulkan | cpu
  layers: -1               # -1 = all layers on GPU
inference:
  model: qwen2.5-7b        # registry key or path to .gguf
  port: 8080
  context_size: 8192
agent:
  type: none                # none | claude-code | custom
  command: ""
policy:
  base: default
  presets: []
machine:
  provider: libkrun
  cpus: 8
  memory: 61440             # MiB (krunkit max ~60 GB)
  disk: 100                 # GB
```

Models cached at `~/.metalclaw/models/`. Metal server built at `~/.metalclaw/bin/`.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  macOS Host                                      │
│                                                  │
│  metalclaw CLI (Python)                          │
│       │                                          │
│  ┌────┴───────────────────────────────────────┐  │
│  │  Metal Mode (default)                      │  │
│  │                                            │  │
│  │  llama-server (Metal GPU)                  │  │
│  │    └── 127.0.0.1:8080 OpenAI-compatible   │  │
│  └────────────────────────────────────────────┘  │
│       │                                          │
│  Podman + libkrun (micro-VM)                     │
│  ┌────────────────────────────────────────────┐  │
│  │  Fedora 40 Container (sandbox)            │  │
│  │                                            │  │
│  │  Agent / workspace (read-only root)       │  │
│  │    └── connects to host inference API     │  │
│  │                                            │  │
│  │  Isolation:                               │  │
│  │    ├── Hypervisor (Apple HV.framework)    │  │
│  │    ├── Filesystem (--read-only + tmpfs)   │  │
│  │    ├── Network (deny-all + presets)       │  │
│  │    └── Process (sandbox user, pids limit) │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Isolation model:**
- **Hypervisor** -- libkrun micro-VM via Apple Hypervisor.framework (stronger than Linux namespaces)
- **Filesystem** -- read-only container root, writable /sandbox and /tmp only
- **Network** -- deny-all default, composable preset allowlists with binary/method restrictions
- **Process** -- non-root sandbox user, PID limit enforcement (4096)

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
│   ├── metal.py        # Host-side Metal inference server
│   ├── models.py       # GGUF model registry and download
│   ├── inference.py    # llama-server health checks
│   ├── policy.py       # Network policy parsing (NemoClaw-compatible)
│   ├── agent.py        # Agent runtime config
│   └── policies/       # Bundled policy YAML files
├── container/
│   ├── Containerfile   # Fedora 40 + Vulkan/RPC + llama.cpp
│   ├── metalclaw-start.sh
│   └── agent-wrapper.sh
├── policies/           # Source policy files
│   ├── default.yaml
│   └── presets/
└── tests/
```

## License

MIT
