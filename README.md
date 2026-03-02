# FactoryMind AI

Enterprise LLM inference platform for industrial environments. On-premise ChatGPT/Perplexity for manufacturing — terminal-first, GPU-efficient, model-swappable.

---

## Quick Start

### 1. Environment

```bash
cp .env.example .env
# Edit .env: set SECRET_KEY (min 32 chars) and adjust paths if needed
```

### 2. Start the server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8080
```

The server will warm up the default model (`llama-3.1-8b-awq`) before accepting traffic. First start downloads model weights (~6 GB) — subsequent starts are instant.

### 3. Start the client (first run)

```bash
python -m client.main
```

On first run a setup wizard will ask for the server URL and your credentials. After that, the TUI launches automatically.

### 4. CLI commands

```bash
# Register / login
python -m client.main register
python -m client.main login

# Interactive chat
python -m client.main chat

# One-shot prompt
python -m client.main chat -m "Explain alarm code E107"

# Use a specific model
python -m client.main chat --model mistral-7b-awq

# List available models
python -m client.main models

# View chat history
python -m client.main history

# Launch full TUI
python -m client.main tui
```

---

## Requirements

- Python 3.12 (managed with `uv`)
- NVIDIA GPU with CUDA (L4 / L40S recommended, ≥ 6 GB VRAM for AWQ models)
- `vllm`, `fastapi`, `uvicorn`, `sqlmodel`, `typer`, `rich`, `httpx`, `textual`

Install all dependencies:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install -r pyproject.toml  # or: uv pip install -e .
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/factorymind.db` | DB connection string |
| `SECRET_KEY` | *(required)* | Min 32-char random string |
| `VLLM_PORT` | `8000` | Internal port for the vLLM process |
| `HF_HOME` | `/cache/huggingface` | HuggingFace model cache |
| `DEFAULT_MODEL` | `llama-3.1-8b-awq` | Model loaded into GPU at startup |
| `PREFETCH_ALL_MODELS` | `true` | Download all catalog weights to disk at startup |
| `ENABLE_REGISTRATION` | `true` | Allow new user sign-ups |
| `MAX_CHAT_HISTORY` | `100` | Messages kept per chat session |

---

## Model Catalog

| Key | Repo | VRAM | Notes |
|---|---|---|---|
| `llama-3.1-8b-instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | ~16 GB | Requires HF token |
| `llama-3.1-8b-awq` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | ~6.5 GB | Default, no token needed |
| `mistral-7b-awq` | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | ~6.5 GB | Good reasoning |
| `qwen-2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | ~16 GB | Multilingual |

Switch models at runtime:

```bash
python -m client.main models          # list + status
# or inside the chat REPL:
/switch mistral-7b-awq
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

No GPU required — the test suite mocks the vLLM manager.

---

## Project Structure

```
factorymind/
├── server/
│   ├── main.py              # FastAPI app + lifespan (warm-up + prefetch)
│   ├── config.py            # Settings via pydantic-settings
│   ├── routers/             # auth, chat, models, history, health
│   ├── services/
│   │   ├── vllm_manager.py  # vLLM process manager + MODEL_CATALOG
│   │   └── chat_service.py  # Chat & history business logic
│   └── models.py            # SQLModel entities
├── client/
│   ├── main.py              # fmind CLI entrypoint + first-run wizard
│   ├── api.py               # APIClient (sync + async streaming)
│   ├── tui.py               # Textual TUI (MainMenu, Chat, Models, History)
│   └── commands/            # chat, models, history, auth
├── tests/
├── .env.example
└── pyproject.toml
```
