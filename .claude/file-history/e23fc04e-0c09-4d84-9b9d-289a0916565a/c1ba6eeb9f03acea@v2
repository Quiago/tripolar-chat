"""POST /v1/chat/completions – OpenAI-compatible, supports streaming SSE."""

import json
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

from server.config import settings
from server.database import engine, get_db
from server.models import Chat, ChatCompletionRequest, User
from server.routers.auth import get_current_user
from server.services.chat_service import create_chat, save_message
from server.services.vllm_manager import MODEL_CATALOG, get_vllm_manager

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """OpenAI-compatible chat completions with auto model loading.

    When stream=True the response is an SSE stream; otherwise a JSON object.
    If the requested model is not loaded the server loads it first (may take
    several minutes on first download) before returning the first token.
    """
    if request.model not in MODEL_CATALOG:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' is not in the catalog. "
                f"Available: {list(MODEL_CATALOG)}"
            ),
        )

    manager = get_vllm_manager()

    # Auto-load / swap model (blocking in thread pool so event loop stays free)
    try:
        await run_in_threadpool(manager.ensure_loaded, request.model)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model '{request.model}': {exc}",
        )

    messages = [m.model_dump() for m in request.messages]
    model_repo = MODEL_CATALOG[request.model].repo

    # Create or resume chat session
    if request.chat_id:
        chat = db.get(Chat, request.chat_id)
        if not chat or chat.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Chat session not found.")
    else:
        first_msg = next(
            (m["content"] for m in messages if m["role"] == "user"),
            "New conversation",
        )
        chat = create_chat(db, current_user.id, request.model, first_msg)

    # Persist user messages
    for msg in messages:
        if msg["role"] == "user":
            save_message(db, chat.id, "user", msg["content"])

    vllm_payload = {
        "model": model_repo,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream,
    }

    if request.stream:
        return StreamingResponse(
            _stream_and_save(vllm_payload, chat.id, settings.vllm_port),
            media_type="text/event-stream",
            headers={"X-Chat-ID": chat.id},
        )

    # ── non-streaming path ────────────────────────────────────────────────────
    try:
        result = await manager.aforward_chat(
            messages,
            request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vLLM error: {exc}")

    assistant_content = result["choices"][0]["message"]["content"]
    save_message(db, chat.id, "assistant", assistant_content)
    result["chat_id"] = chat.id
    return result


async def _stream_and_save(
    payload: dict,
    chat_id: str,
    vllm_port: int,
) -> AsyncGenerator[str, None]:
    """Yield SSE lines from vLLM and persist the full response when done."""
    content_parts: list[str] = []

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"http://localhost:{vllm_port}/v1/chat/completions",
                json=payload,
                timeout=300.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    yield line + "\n\n"
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            data = json.loads(line[6:])
                            delta = (
                                data["choices"][0]["delta"].get("content") or ""
                            )
                            if delta:
                                content_parts.append(delta)
                        except Exception:
                            pass
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        # Save assistant reply using a fresh session (request session is gone)
        if content_parts:
            with Session(engine) as db:
                save_message(db, chat_id, "assistant", "".join(content_parts))
