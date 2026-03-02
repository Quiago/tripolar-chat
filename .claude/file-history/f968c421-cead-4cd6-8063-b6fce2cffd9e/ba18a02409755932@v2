from sqlmodel import Session, select

from server.core.exceptions import (
    AuthError,
    DuplicateUserError,
    RegistrationDisabledError,
)
from server.core.security import generate_api_key, hash_password, verify_password
from server.models import RegisterRequest, User


def register_user(req: RegisterRequest, db: Session, *, registration_enabled: bool) -> User:
    """Create a new user. Raises DuplicateUserError or RegistrationDisabledError."""
    if not registration_enabled:
        raise RegistrationDisabledError("New user registration is disabled.")

    if db.exec(select(User).where(User.username == req.username)).first():
        raise DuplicateUserError(f"Username '{req.username}' is already taken.")

    if db.exec(select(User).where(User.email == req.email)).first():
        raise DuplicateUserError(f"Email '{req.email}' is already registered.")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(username: str, password: str, db: Session) -> User:
    """Return user if credentials are valid, else raise AuthError."""
    user = db.exec(select(User).where(User.username == username)).first()
    if not user or not verify_password(password, user.hashed_password):
        raise AuthError("Invalid username or password.")
    if not user.is_active:
        raise AuthError("Account is deactivated.")
    return user


def get_user_by_api_key(api_key: str, db: Session) -> User:
    """Return the user that owns this API key, or raise AuthError."""
    user = db.exec(select(User).where(User.api_key == api_key)).first()
    if not user:
        raise AuthError("Invalid API key.")
    if not user.is_active:
        raise AuthError("Account is deactivated.")
    return user


def rotate_api_key(user: User, db: Session) -> User:
    """Generate and persist a new API key for the user."""
    user.api_key = generate_api_key()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
