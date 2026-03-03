"""POST /v1/chat/completions – OpenAI-compatible, supports streaming SSE."""

import json
import logging
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool

log = logging.getLogger(__name__)

from server.config import settings
from server.database import engine, get_db
from server.models import Chat, ChatCompletionRequest, User
from server.routers.auth import get_current_user
from server.services.chat_service import create_chat, save_message
from server.services.tools import TOOLS, build_connector_context, run_tool_loop
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
        # Log the full error on the server; return a brief message to the client.
        log.error("Failed to load model '%s': %s", request.model, exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model '{request.model}'. Check server logs.",
        )

    messages = [m.model_dump() for m in request.messages]
    model_repo = MODEL_CATALOG[request.model].repo

    # ── Inject connector context into system prompt ────────────────────────────
    context = build_connector_context(db, user_id=current_user.id)
    system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    if system_idx is not None:
        messages[system_idx]["content"] += f"\n\n{context}"
    else:
        messages.insert(0, {"role": "system", "content": context})

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
        "tools": TOOLS,
        "tool_choice": "auto",
    }

    if request.stream:
        return StreamingResponse(
            _stream_and_save(
                vllm_payload, chat.id, settings.vllm_port,
                user_id=current_user.id, db=db,
            ),
            media_type="text/event-stream",
            headers={"X-Chat-ID": chat.id},
        )

    # ── non-streaming path (includes tool-call loop) ──────────────────────────
    async def _vllm(msgs: list[dict]) -> dict:
        return await manager.aforward_chat(
            msgs, request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            tools=TOOLS,
            tool_choice="auto",
        )

    try:
        result = await _vllm(messages)
        result = await run_tool_loop(result, messages, current_user.id, db, vllm_call=_vllm)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vLLM error: {exc}")

    assistant_content = result["choices"][0]["message"].get("content") or ""
    save_message(db, chat.id, "assistant", assistant_content)
    result["chat_id"] = chat.id
    return result


async def _stream_and_save(
    payload: dict,
    chat_id: str,
    vllm_port: int,
    user_id: str = "",
    db=None,
) -> AsyncGenerator[str, None]:
    """Yield SSE lines from vLLM, handle tool_calls mid-stream, persist reply.

    When the model emits finish_reason=tool_calls in the stream:
      1. A status line is sent to the client ("[Querying connector…]")
      2. Tools are executed
      3. A new non-streaming vLLM request is made with tool results
      4. The final answer is streamed back token-by-token
    """
    content_parts: list[str] = []
    # Accumulate tool call deltas from the stream
    tool_call_accum: dict[int, dict] = {}  # index → {id, name, arguments}
    finish_reason: str = ""

    # ── Phase 1: stream and detect tool_calls ────────────────────────────────
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
                    if not line.startswith("data: ") or "[DONE]" in line:
                        yield line + "\n\n"
                        continue

                    try:
                        data = json.loads(line[6:])
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason

                        # Accumulate tool call chunks
                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in tool_call_accum:
                                tool_call_accum[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_call_accum[idx]["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tool_call_accum[idx]["function"]["arguments"] += fn["arguments"]

                        # Accumulate normal text content
                        text = delta.get("content") or ""
                        if text:
                            content_parts.append(text)
                            yield line + "\n\n"
                        elif not delta.get("tool_calls"):
                            # Pass through non-tool, non-content lines (role, etc.)
                            yield line + "\n\n"

                    except Exception:
                        yield line + "\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    # ── Phase 2: execute tool calls if any ──────────────────────────────────
    if finish_reason == "tool_calls" and tool_call_accum:
        tool_calls = [tool_call_accum[i] for i in sorted(tool_call_accum)]

        # Send a status event so the UI shows progress
        names = ", ".join(tc["function"]["name"] for tc in tool_calls)
        status_chunk = {
            "choices": [{
                "delta": {"content": f"\n*[Querying: {names}…]*\n\n"},
                "finish_reason": None,
            }]
        }
        yield f"data: {json.dumps(status_chunk)}\n\n"

        # Build messages with tool results
        messages = list(payload.get("messages", []))
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })

        with Session(engine) as tool_db:
            for tc in tool_calls:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                from server.services.tools import execute_tool
                result_str = execute_tool(
                    fn["name"], args, user_id=user_id, db=tool_db
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        # ── Phase 3: re-submit and stream the final answer ───────────────────
        final_payload = {**payload, "messages": messages, "tools": TOOLS}
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"http://localhost:{vllm_port}/v1/chat/completions",
                    json=final_payload,
                    timeout=300.0,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        yield line + "\n\n"
                        if line.startswith("data: ") and "[DONE]" not in line:
                            try:
                                data = json.loads(line[6:])
                                text = data["choices"][0]["delta"].get("content") or ""
                                if text:
                                    content_parts.append(text)
                            except Exception:
                                pass
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    # ── Persist ──────────────────────────────────────────────────────────────
    if content_parts:
        with Session(engine) as db:
            save_message(db, chat_id, "assistant", "".join(content_parts))
