from datetime import datetime
from typing import Optional
import uuid

from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    username: str = Field(index=True, unique=True, min_length=3, max_length=64)
    email: str = Field(unique=True)
    hashed_password: str
    api_key: str = Field(default_factory=lambda: str(uuid.uuid4()), unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class Chat(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    title: str = Field(default="New conversation")
    model_used: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    chat_id: str = Field(foreign_key="chat.id", index=True)
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Request / Response schemas ────────────────────────────────────────────────

class RegisterRequest(SQLModel):
    username: str
    email: str
    password: str


class LoginRequest(SQLModel):
    username: str
    password: str


class UserPublic(SQLModel):
    """Safe representation – never exposes hashed_password."""
    id: str
    username: str
    email: str
    api_key: str
    created_at: datetime
    is_active: bool
