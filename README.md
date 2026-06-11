<div align="center">

# 👻 OpenCode Zen Gateway

**Proxy gateway for OpenCode API**

Made with ❤️ by [@AsimAftab](https://github.com/AsimAftab)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

*Use OpenCode models natively with Claude Code, OpenClaw, Claw Code, Codex app, Cursor, Cline, Roo Code, LangChain, and other OpenAI or Anthropic compatible tools*

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔌 **OpenAI-compatible API** | Works with any OpenAI-compatible tool |
| 🔌 **Anthropic-compatible API** | Native `/v1/messages` endpoint |
| 🧠 **Extended Thinking** | Fake reasoning injection to mimic thinking tags |
| 🔍 **Web Search** | Auto-injected web search capabilities |
| 🛠️ **Tool Calling** | Supports full function calling and MCP tools |
| 💬 **Full message history** | Passes complete conversation context |
| 📡 **Streaming** | Full SSE streaming support |
| 📋 **Dynamic models list** | Automatically fetches all available OpenCode models |

---

## 🤖 Available Free Models

The gateway dynamically fetches the complete list of available models directly from the OpenCode API. Currently, OpenCode offers several high-tier models completely for free (you can generally identify them by the `-free` suffix).

Some of the currently available **free models** include:
- `deepseek-v4-flash-free` — DeepSeek v4 Flash Free OpenCode Zen
- `nemotron-3-ultra-free` — Nemotron 3 Ultra Free
- `mimo-v2.5-free` — Mimo v2.5 Free
- `north-mini-code-free` — North Mini Code Free
- `big-pickle` — Big Pickle (Free Tier)
- `qwen3.6-plus-free` — Qwen 3.6 Plus Free
- `minimax-m3-free` — Minimax M3 Free

### How to find and use them
Because the gateway natively proxies the `/v1/models` endpoint, Claude Code will automatically fetch the live list of all available models from OpenCode.

You can simply start Claude Code and use its interactive model selector to see all the latest free models, or launch Claude Code directly with your desired free model:
```bash
claude -m deepseek-v4-flash-free
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An [OpenCode](https://opencode.ai) API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/AsimAftab/opencode-zen-gateway.git
cd opencode-zen-gateway

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
```

### Configuration

Edit your `.env` file and add your actual OpenCode API key and a custom password to protect your gateway:

```env
OPENCODE_API_KEY="your_actual_opencode_api_key_here"

# Password to protect YOUR proxy server
# You'll use this as api_key when connecting to your gateway in Claude Code
PROXY_API_KEY="my-super-secret-password-123"
```

### Start the Server

```bash
python main.py
```

The server will be available at `http://localhost:8000`

---

## 🔌 Usage in Claude Code

### Using Anthropic Format
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="my-super-secret-password-123"
claude
```

### Using OpenAI Format
```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="my-super-secret-password-123"
claude
```

## 🛠 Advanced Features

### Truncation Recovery
The gateway automatically detects Anthropic context truncation limits and injects synthetic user/tool messages to preserve conversation state across extremely long context windows.

### Fake Reasoning
We intercept reasoning configs and format them natively back to the client as `<thinking>` blocks so applications expecting Anthropic thinking behavior still work flawlessly.
