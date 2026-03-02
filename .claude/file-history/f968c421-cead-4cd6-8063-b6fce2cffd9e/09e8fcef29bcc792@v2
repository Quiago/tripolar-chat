"""Business logic for chat sessions and message history."""

from datetime import datetime
from typing import List, Optional, Tuple

from sqlmodel import Session, select

from server.models import Chat, Message


def create_chat(
    db: Session,
    user_id: str,
    model_name: str,
    first_user_message: str,
) -> Chat:
    """Create and persist a new Chat, titling it from the first message."""
    title = first_user_message[:60] + ("…" if len(first_user_message) > 60 else "")
    chat = Chat(user_id=user_id, title=title, model_used=model_name)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def save_message(db: Session, chat_id: str, role: str, content: str) -> Message:
    """Persist a message and bump the parent Chat's updated_at timestamp."""
    msg = Message(chat_id=chat_id, role=role, content=content)
    db.add(msg)

    chat = db.get(Chat, chat_id)
    if chat:
        chat.updated_at = datetime.utcnow()
        db.add(chat)

    db.commit()
    db.refresh(msg)
    return msg


def get_user_chats(db: Session, user_id: str, limit: int = 20) -> List[Chat]:
    """Return the user's most-recently-updated chats, newest first."""
    return list(
        db.exec(
            select(Chat)
            .where(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc())
            .limit(limit)
        ).all()
    )


def get_chat_messages(
    db: Session,
    chat_id: str,
    user_id: str,
) -> Optional[Tuple[Chat, List[Message]]]:
    """Return (chat, messages) or None if not found / not owned by user."""
    chat = db.get(Chat, chat_id)
    if not chat or chat.user_id != user_id:
        return None
    messages = list(
        db.exec(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at)
        ).all()
    )
    return chat, messages
