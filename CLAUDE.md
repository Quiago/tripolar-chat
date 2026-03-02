CLAUDE.md – FactoryMind AI PlatformContext for Developers and AI AssistantsProject VisionFactoryMind AI is an enterprise LLM inference platform for industrial environments (manufacturing). Conceptually, it is an on‑premise ChatGPT/Perplexity where:Users connect via CLI (terminal) or API.They can select different open‑source models (Llama, Mistral, Qwen, etc.).The system loads only one model into GPU memory at a time, and swaps models on demand.The server stores user profiles, API keys, and chat histories, and will later execute tools/connectors (e.g., OPC UA, databases, ERPs).The initial deployment target is a GPU server running in Lightning, but the design must be portable to on‑premise bare‑metal or Kubernetes (k8s).High‑Level ArchitectureCLIENT (Terminal / CLI)
┌──────────────────────────────────────────────────────────────┐
│ fmind CLI (Python + Typer + Rich + HTTPX)                    │
│  - Auth via API key                                          │
│  - Commands: login, chat, models, history                    │
└──────────────────────────────────────────────────────────────┘
                │
                │ HTTPS / (later WSS for streaming)
                ▼
SERVER (Lightning AI → later On-Prem)
┌──────────────────────────────────────────────────────────────┐
│  FastAPI API Server                                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - Auth (API keys, later JWT)                           │  │
│  │ - User & API key management                            │  │
│  │ - Chat sessions & message history                      │  │
│  │ - Rate limiting hooks (per user)                       │  │
│  │ - Forwards chat requests to the Model Manager          │  │
│  └────────────────────────────────────────────────────────┘  │
│                ▲                                             │
│                │                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ vLLM Model Manager                                     │  │
│  │  - Single vLLM process (OpenAI-compatible server)      │  │
│  │  - Loads exactly ONE model at a time                   │  │
│  │  - Swaps models on demand (stop + start)               │  │
│  │  - Uses persistent Hugging Face cache                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Persistent Storage                                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ PostgreSQL (prod) / SQLite (dev)                       │  │
│  │  - users: profiles, hashed passwords, api_keys         │  │
│  │  - chats: conversations, metadata (model used, title)  │  │
│  │  - messages: full history per chat                     │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ File System Volumes                                    │  │
│  │  - /cache/huggingface/  → HF model cache (persisted)   │  │
│  │  - /data/               → DB file in dev (SQLite)      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
Technology StackCorePython 3.12 (managed with uv).vLLM as the inference engine and OpenAI‑compatible server.FastAPI for the HTTP API (async, OpenAPI docs auto‑generated).SQLModel (SQLAlchemy + Pydantic) for strongly typed DB models.ClientTyper for the CLI (declarative commands with type hints).Rich for terminal UX (colors, markdown, progress bars).HTTPX as an async HTTP client.DatabasesSQLite for development and local testing.PostgreSQL for production/on‑prem.Environment / Infrauv to manage Python environments and dependencies.Lightning AI GPU instance (L4) for development/MVP.Docker + Docker Compose for on‑prem deployment later.NVIDIA GPU stack (CUDA + driver) available on the server.Python Environment (uv – Required)The project must use uv to manage environments and dependencies, not raw pip.Already executed for the current env:uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install fastapi uvicorn requests psutil
You MUST also install the following (if not yet installed):uv pip install sqlmodel typer rich httpx
uv pip install python-jose[cryptography] passlib[bcrypt]  # Auth support
uv pip install pydantic-settings
Rules:Do not hardcode paths, secrets, ports, or tokens.All configuration must be read via environment variables and/or a .env file, using pydantic-settings.Project Structurefactorymind/
├── CLAUDE.md                      # This file
├── README.md                      # User-facing setup + usage
├── docker-compose.yml             # API + DB + volumes (later)
├── .env.example                   # Template for environment variables
├── pyproject.toml                 # Project metadata / dependencies (uv-compatible)
├── server/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entrypoint
│   ├── config.py                  # Settings via Pydantic BaseSettings
│   ├── auth.py                    # Auth utils (API keys, JWT later)
│   ├── models.py                  # SQLModel entities (User, Chat, Message)
│   ├── database.py                # DB engine + SessionLocal + get_db()
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py                # /auth/* endpoints
│   │   ├── chat.py                # /v1/chat/completions, /v1/history
│   │   ├── models.py              # /v1/models
│   │   └── health.py              # /health
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vllm_manager.py        # vLLM process + model swap logic
│   │   ├── chat_service.py        # Chat & history business logic
│   │   └── user_service.py        # Users & API key management
│   └── core/
│       ├── __init__.py
│       ├── security.py            # Password hashing, API key/JWT helpers
│       └── exceptions.py          # Custom exception hierarchy
├── client/
│   ├── __init__.py
│   ├── main.py                    # `fmind` Typer entrypoint
│   ├── config.py                  # Local client config (~/.fmind/config.yaml)
│   ├── api.py                     # HTTP client wrapper around server API
│   └── commands/
│       ├── __init__.py
│       ├── auth.py                # fmind login/logout
│       ├── chat.py                # fmind chat
│       ├── models.py              # fmind models
│       └── history.py             # fmind history
└── tests/
    ├── conftest.py
    ├── test_api.py
    └── test_vllm_manager.py
Design Principles (Must Follow)1. No Hard‑Coded ValuesNever hardcode: URLs, ports, DB connection strings, secrets, tokens, API keys, or filesystem paths. Use pydantic-settings and environment variables.Example (correct):# server/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/factorymind.db"
    secret_key: str  # required, no default – must be set in .env
    api_key_header: str = "X-API-Key"
    vllm_port: int = 8000
    hf_cache_dir: str = "/cache/huggingface"

    class Config:
        env_file = ".env"

settings = Settings()
All other modules must import from settings, not from os.environ directly (except inside config.py).2. Separation of ConcernsRouters (server/routers/*.py): Validate input/output. Handle HTTP details (status codes, headers). Delegate to services.Services (server/services/*.py): Business logic only. Coordinate DB operations and vLLM calls.Core (server/core/*.py): Cross‑cutting concerns (security, exceptions, helpers).Database: server/database.py handles engine and sessions.3. Strong Typing EverywhereUse type hints for all functions, methods, and class attributes.Use SQLModel for DB models (inherits from Pydantic, typed).Use Pydantic models for request/response schemas in FastAPI.4. Async‑First BackendAll FastAPI endpoints should be async def.Use httpx.AsyncClient for internal HTTP calls (if needed).DB access can initially be sync with SessionLocal; later, we can migrate to async session.5. Structured Error HandlingDefine a small hierarchy of exceptions:# server/core/exceptions.py
class FactoryMindError(Exception):
    """Base exception for domain errors."""
    pass

class ModelNotAvailableError(FactoryMindError):
    """Requested model is not in the catalog or not allowed."""
    pass

class VLLMStartupError(FactoryMindError):
    """vLLM failed to start."""
    pass
In routers, catch domain errors and map them to appropriate HTTP status codes:from fastapi import HTTPException
from server.core.exceptions import ModelNotAvailableError

try:
    # Service call here
    pass
except ModelNotAvailableError as e:
    raise HTTPException(status_code=400, detail=str(e))
vLLM Model Manager – Core BehaviorWe run one vLLM server process (OpenAI‑compatible) at a time, bound to a single GPU (L4/L40S/etc). We keep a model catalog and implement hot‑swap.Catalog (centralized, no magic strings)# server/services/vllm_manager.py (sketch)
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class ModelConfig:
    repo: str
    quantization: Optional[str]
    tensor_parallel: int
    gpu_memory_util: float
    max_model_len: int
    description: str

MODEL_CATALOG: Dict[str, ModelConfig] = {
    "llama-3.1-8b": ModelConfig(
        repo="meta-llama/Llama-3.1-8B-Instruct",
        quantization=None,
        tensor_parallel=1,
        gpu_memory_util=0.85,
        max_model_len=8192,
        description="Meta Llama 3.1 8B - general purpose"
    ),
    "llama-3.1-8b-awq": ModelConfig(
        repo="casperhansen/llama-3.1-8b-instruct-awq",
        quantization="awq",
        tensor_parallel=1,
        gpu_memory_util=0.90,
        max_model_len=8192,
        description="Llama 3.1 8B 4-bit AWQ - VRAM-efficient"
    ),
    "mistral-7b-awq": ModelConfig(
        repo="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        quantization="awq",
        tensor_parallel=1,
        gpu_memory_util=0.90,
        max_model_len=8192,
        description="Mistral 7B - good reasoning, compact"
    ),
    "qwen-2.5-7b": ModelConfig(
        repo="Qwen/Qwen2.5-7B-Instruct",
        quantization=None,
        tensor_parallel=1,
        gpu_memory_util=0.85,
        max_model_len=8192,
        description="Qwen 2.5 7B - multilingual"
    ),
}
Manager ResponsibilitiesKeep track of: current_model: Optional[str] and process: Optional[subprocess.Popen].Start/stop the vLLM process with the correct arguments.Ensure the HF cache is persistent and reused:HF_HOME=/cache/huggingfaceTRANSFORMERS_CACHE=/cache/huggingfaceProvide methods:ensure_loaded(model_name: str) -> Noneforward_chat(messages: List[dict], **params) -> dictis_ready() -> boolis_cached(model_name: str) -> bool(Note: The manager must not import FastAPI; it is a pure Python service.)API DesignThe server exposes an OpenAI‑compatible API as much as possible.AuthPrimary auth: API key via header, e.g., X-API-Key: <key>.User Model:from sqlmodel import SQLModel, Field
from datetime import datetime
import uuid

class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(unique=True)
    hashed_password: str
    api_key: str = Field(default_factory=lambda: str(uuid.uuid4()), unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
Key Endpoints1. Health (GET /health)Returns the status of the API and vLLM:{
  "status": "healthy",
  "model_manager": {
    "current_model": "llama-3.1-8b-awq",
    "is_ready": true,
    "catalog": ["llama-3.1-8b", "llama-3.1-8b-awq", "mistral-7b-awq", "qwen-2.5-7b"]
  },
  "timestamp": "2026-03-02T20:48:36Z"
}
2. List Models (GET /v1/models)Requires API key.Returns a subset of the catalog, including whether the model is cached and/or loaded.3. Chat Completions (POST /v1/chat/completions)Requires API key.Body (OpenAI‑style):{
  "model": "llama-3.1-8b-awq",
  "messages": [
    {"role": "system", "content": "You are an assistant for factory operators."},
    {"role": "user", "content": "What is the normal temperature range for Motor A?"}
  ],
  "max_tokens": 256,
  "temperature": 0.2
}
Behavior:Validate that the model is in MODEL_CATALOG.Call vllm_manager.ensure_loaded(model).Forward the request to the vLLM server (/v1/chat/completions).Persist user+assistant messages in the DB (Chat + Message tables).Return the vLLM response unwrapped or in a lightly adapted OpenAI format.CLI Design (Client Side)We will build a fmind CLI that behaves like:# Configure server + API key
fmind login --server https://<lightning-url> --api-key <key>

# List models
fmind models

# Start a chat (interactive)
fmind chat --model llama-3.1-8b-awq

# Or one-shot prompt
fmind chat --model mistral-7b-awq -m "Explain this alarm code: E107"
Implementation Details:Use Typer for commands and subcommands.Store config under ~/.fmind/config.yaml (server URL, API key, default model).Use HTTPX to talk to the API.Use Rich for colored roles (user vs assistant), simple markdown rendering, and spinners during model load.Environment Variables (.env)Create .env from .env.example:# Database
DATABASE_URL=sqlite:///./data/factorymind.db
# For prod:
# DATABASE_URL=postgresql://user:pass@db-host/factorymind

# Security
SECRET_KEY=change-me-in-production-min-32-chars
API_KEY_HEADER=X-API-Key

# vLLM
VLLM_PORT=8000
HF_HOME=/cache/huggingface
TRANSFORMERS_CACHE=/cache/huggingface
CUDA_VISIBLE_DEVICES=0

# Features
ENABLE_REGISTRATION=true
MAX_CHAT_HISTORY=100
No secret or config should be hard‑coded in the code.Development Commands# Assumes: uv venv + packages already installed

# Run backend locally
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# (Later) Run tests
pytest tests/ -v

# Install and run CLI (locally)
cd client
pip install -e .
fmind --help
Decisions Already Made (Do Not Change Without Discussion)vLLM is the primary inference layer (not TensorRT‑LLM) for flexibility with Hugging Face models and rapid iteration.Single‑model‑in‑GPU policy per server instance; we use model swapping because target GPUs (e.g., L4) have limited VRAM.SQLite for dev, PostgreSQL for prod.CLI‑first UX, web UI will come later if needed.OpenAI‑compatible API shape to simplify integration and testing.Short‑Term RoadmapWeek 1: Implement VLLMModelManager and run it on Lightning. Manual API to swap models and basic /v1/chat/completions.Week 2: Add user/auth layer (API keys), DB persistence, and /v1/models.Week 3: Implement fmind CLI (login, models, chat).Week 4: Persist chat sessions and basic profile behavior (list history).Week 5: Add Docker Compose for on‑premise deployment.Notes for Claude (and Other AI Assistants)Always ask for clarification if architecture or constraints are ambiguous.Never hardcode secrets, tokens, or environment‑specific paths.Always use type hints and Pydantic/SQLModel models.Write tests for: model swap behavior, auth and API key checks, and basic chat flow.Use clear docstrings (Google or NumPy style) for non‑trivial functions.Maintain compatibility with Python 3.12 and uv‑driven workflows.ReferencesvLLM Quickstart (OpenAI-compatible server, one model at a time)