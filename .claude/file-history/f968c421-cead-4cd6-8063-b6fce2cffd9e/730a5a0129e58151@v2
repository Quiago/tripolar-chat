"""GET /v1/history – list and retrieve chat sessions with messages."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from server.database import get_db
from server.models import ChatDetailPublic, ChatPublic, MessagePublic, User
from server.routers.auth import get_current_user
from server.services.chat_service import get_chat_messages, get_user_chats

router = APIRouter(prefix="/v1/history", tags=["history"])


@router.get("", response_model=List[ChatPublic])
def list_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the user's most-recently-updated chat sessions."""
    chats = get_user_chats(db, current_user.id, limit=limit)
    return [
        ChatPublic(
            id=c.id,
            title=c.title,
            model_used=c.model_used,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in chats
    ]


@router.get("/{chat_id}", response_model=ChatDetailPublic)
def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a single chat session with its full message history."""
    result = get_chat_messages(db, chat_id, current_user.id)
    if result is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    chat, messages = result
    return ChatDetailPublic(
        id=chat.id,
        title=chat.title,
        model_used=chat.model_used,
        messages=[
            MessagePublic(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )
