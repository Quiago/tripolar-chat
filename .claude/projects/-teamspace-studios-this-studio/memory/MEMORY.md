# FactoryMind AI Platform – Project Memory

## What this project is
Enterprise CLI + API for on-premise LLM inference (vLLM backend), targeting industrial/manufacturing. Think "on-prem ChatGPT with a terminal client."

## Current state (completed)
- Auth: register, login, logout, whoami, rotate-key ✓
- Server: FastAPI + SQLModel + SQLite/Postgres ✓
- Chat: POST /v1/chat/completions (streaming SSE + non-streaming) ✓
- Models: GET /v1/models, POST /v1/models/{name}/load ✓
- History: GET /v1/history, GET /v1/history/{id} ✓
- CLI: fmind chat (interactive + one-shot), fmind models, fmind history ✓
- TUI: Textual-based TUI (MainMenu, ChatScreen, ModelSelect, HistoryList) ✓
- First-run setup wizard in fmind with no args ✓

## Key architecture
- `server/services/vllm_manager.py` – singleton VLLMModelManager, MODEL_CATALOG dataclass
- `client/api.py` – APIClient class (sync chat_stream + async async_chat_stream for TUI)
- Auth via X-API-Key header; `get_current_user()` in `server/routers/auth.py`
- Streaming: server returns StreamingResponse with SSE, saves to DB in finally block with new Session(engine)
- Model auto-load: server calls ensure_loaded via run_in_threadpool (blocks until ready)
- Textual 8.x installed (>=0.50 specified); @work decorator for async workers

## bcrypt issue
- passlib 1.7.x incompatible with bcrypt>=4 → pin `bcrypt<4.0` in pyproject.toml

## Run server
```bash
cp .env.example .env  # edit SECRET_KEY
uvicorn server.main:app --reload --host 0.0.0.0 --port 8080
```

## Testing
```bash
SECRET_KEY="test-secret-key-32-chars-minimum!!" python -c "from fastapi.testclient import TestClient; ..."
```
Uses TestClient with `with` statement to trigger lifespan (creates DB tables).
