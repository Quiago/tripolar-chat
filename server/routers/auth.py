from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlmodel import Session

from server.config import settings
from server.core.exceptions import AuthError, DuplicateUserError, RegistrationDisabledError
from server.database import get_db
from server.models import LoginRequest, RegisterRequest, User, UserPublic
from server.services.user_service import (
    authenticate_user,
    get_user_by_api_key,
    register_user,
    rotate_api_key,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_api_key_scheme = APIKeyHeader(name=settings.api_key_header, auto_error=True)


def get_current_user(
    api_key: str = Security(_api_key_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        return get_user_by_api_key(api_key, db)
    except AuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/register", response_model=UserPublic, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account and return the API key."""
    try:
        user = register_user(req, db, registration_enabled=settings.enable_registration)
    except RegistrationDisabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except DuplicateUserError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return user


@router.post("/login", response_model=UserPublic)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with username + password and return the API key."""
    try:
        user = authenticate_user(req.username, req.password, db)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return user


@router.get("/me", response_model=UserPublic)
def me(current_user: User = Depends(get_current_user)):
    """Return the profile of the authenticated user."""
    return current_user


@router.post("/rotate-key", response_model=UserPublic)
def rotate_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a new API key for the authenticated user."""
    return rotate_api_key(current_user, db)
