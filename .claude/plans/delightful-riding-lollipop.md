# Plan: chat command, TUI, models, history

## Context
Auth is complete. Now we add the core features: streaming chat, interactive TUI, model listing, history, and a first-run setup wizard. The server needs OpenAI-compatible `/v1/chat/completions` (streaming SSE), `/v1/models`, and `/v1/history` endpoints. The client needs `chat`, `models`, `history`, `tui` commands.

---

## Files to create

### Server (5 new, 3 updated)

| File | Action | Purpose |
|---|---|---|
| `server/services/vllm_manager.py` | CREATE | Singleton vLLM process manager + MODEL_CATALOG |
| `server/services/chat_service.py` | CREATE | create_chat / save_message / get_user_chats / get_chat_messages |
| `server/routers/chat.py` | CREATE | POST /v1/chat/completions (stream + non-stream) |
| `server/routers/models.py` | CREATE | GET /v1/models, POST /v1/models/{name}/load |
| `server/routers/history.py` | CREATE | GET /v1/history, GET /v1/history/{id} |
| `server/models.py` | UPDATE | Add ChatMessageSchema, ChatCompletionRequest, ChatPublic, ChatDetailPublic, MessagePublic |
| `server/main.py` | UPDATE | Include new routers, graceful vLLM shutdown in lifespan |
| `server/routers/health.py` | UPDATE | Include vLLM status from manager |

### Client (4 new, 3 updated)

| File | Action | Purpose |
|---|---|---|
| `client/commands/chat.py` | CREATE | `fmind chat` – interactive + one-shot + streaming |
| `client/commands/models.py` | CREATE | `fmind models` – table + `--interactive` picker |
| `client/commands/history.py` | CREATE | `fmind history` – list + view session |
| `client/tui.py` | CREATE | Textual TUI – MainMenu, ChatScreen, ModelSelect |
| `client/api.py` | UPDATE | Add `APIClient` class with `chat_stream()`, `list_models()`, `list_history()`, `get_chat()` |
| `client/main.py` | UPDATE | Wire commands, first-run wizard, `fmind tui`, `fmind` no-args |
| `pyproject.toml` | UPDATE | Add `textual>=0.50`, `typer[all]` |

---

## Key design decisions

### 1. vLLM Manager singleton
Module-level `_instance` + `get_vllm_manager()` factory in `server/services/vllm_manager.py`. MODEL_CATALOG dataclass identical to CLAUDE.md spec. Methods: `ensure_loaded`, `stop`, `is_ready`, `is_cached`, `get_info`, `aforward_chat`.

### 2. Model loading – auto-load with terminal spinner
Server calls `ensure_loaded` via `run_in_threadpool` (doesn't block event loop). The HTTP request hangs until the model is ready (client timeout = 300 s). Client shows spinner: "Loading llama-3.1-8b… (this may take a few minutes)". Spinner stops when first streaming token arrives.

```python
from starlette.concurrency import run_in_threadpool
await run_in_threadpool(manager.ensure_loaded, request.model)
# then forward to vLLM – first token triggers client spinner to stop
```

### 3. Streaming + DB save
`StreamingResponse` with async generator that:
- Forwards SSE chunks from vLLM with `httpx.AsyncClient.stream()`
- Accumulates full response in a list
- In `finally` block: opens a **new** `Session(engine)` (not the request session, which closes when the endpoint returns) and saves the assistant message

```python
async def _stream_and_save(payload, chat_id, vllm_port):
    content_parts = []
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", ...) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n\n"
                        # accumulate delta
    finally:
        if content_parts:
            with Session(engine) as db:
                save_message(db, chat_id, "assistant", "".join(content_parts))
```

### 4. Chat command UX
```
[llama-3.1-8b] You: <user types here>
                    <spinner while waiting for first token>
[llama-3.1-8b] > Hello! I can help you with...  ← tokens stream in
```
- `/models` → table from server
- `/switch <name>` → update model, clear history
- `/clear` → clear screen + local history
- `/exit`, Ctrl+C → graceful exit
- `fmind chat -m "..."` → one-shot, no loop

### 5. APIClient class (added to api.py, free functions kept)
```python
class APIClient:
    def __init__(self, server_url: str, api_key: str)
    def chat_stream(self, model, messages, **kwargs) -> Iterator[str]
    def chat(self, model, messages, **kwargs) -> dict
    def list_models(self) -> list[dict]
    def list_history(self, limit=20) -> list[dict]
    def get_chat(self, chat_id) -> dict
    def load_model(self, model_name) -> dict
```

`chat_stream` parses SSE: `data: {...}` → yields `choices[0].delta.content`.

### 6. TUI (Textual) – full implementation
Three screens, keyboard navigation, async streaming via `@work` workers:

**MainMenuScreen** – `ListView` of 6 items (Chat, Models, History, Account, Logout, Exit), ↑↓ + Enter.

**ChatScreen** – `RichLog` (scrollable message history) + `Input` at bottom. On Enter: calls `_stream_response` worker which uses `APIClient.async_chat_stream()` (httpx AsyncClient + aiter_lines). Chunks posted to RichLog as they arrive. Shows "[model] You: …" / "[model] > chunk…chunk" pattern.

**ModelSelectScreen** – `DataTable` with model, description, cached/loaded status. Enter sets model in config.

`APIClient` needs both sync `chat_stream` (for CLI) and async `async_chat_stream` (for TUI):
```python
async def async_chat_stream(self, model, messages, **kwargs):
    async with httpx.AsyncClient(...) as c:
        async with c.stream("POST", "/v1/chat/completions", json={..., "stream": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content","")
                    if delta: yield delta
```

App launched by `fmind tui` or `fmind` (no args) if credentials exist.
If Textual not installed → print install hint.

### 7. First-run wizard (in client/main.py callback)
```python
@app.callback(invoke_without_command=True)
def _callback(ctx: typer.Context):
    if ctx.invoked_subcommand is not None:
        return
    if not cfg.get_server_url():
        _run_setup_wizard()
    else:
        _launch_tui()
```
Wizard: prompt server URL → ask if has account → login OR register → save config.

### 8. server/models.py additions
```python
class ChatMessageSchema(SQLModel):
    role: str; content: str

class ChatCompletionRequest(SQLModel):
    model: str
    messages: list[ChatMessageSchema]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    chat_id: Optional[str] = None

class MessagePublic(SQLModel):
    id: str; role: str; content: str; created_at: datetime

class ChatPublic(SQLModel):
    id: str; title: str; model_used: str; created_at: datetime; updated_at: datetime

class ChatDetailPublic(ChatPublic):
    messages: list[MessagePublic]
```

---

## Verification
```bash
# 1. Syntax check all files
python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p) and '.claude' not in str(p)]"

# 2. Import test (no SECRET_KEY in env)
SECRET_KEY="test-secret-key-32-chars-minimum!!" python -c "
from server.main import app
from client.commands.chat import app as chat_app
from client.commands.models import app as models_app
from client.commands.history import app as history_app
from client.tui import FactoryMindTUI
print('All imports OK')
"

# 3. Auth + models + history integration test via TestClient
# (existing auth tests continue to pass)

# 4. Manual test of CLI help
# fmind --help  (shows chat, models, history, tui, login, logout, register, whoami)
```
