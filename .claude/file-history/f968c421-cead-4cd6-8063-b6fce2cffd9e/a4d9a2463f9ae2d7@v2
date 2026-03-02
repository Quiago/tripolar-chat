class FactoryMindError(Exception):
    """Base exception for domain errors."""


class ModelNotAvailableError(FactoryMindError):
    """Requested model is not in the catalog or not allowed."""


class VLLMStartupError(FactoryMindError):
    """vLLM failed to start."""


class AuthError(FactoryMindError):
    """Invalid credentials or API key."""


class RegistrationDisabledError(FactoryMindError):
    """New user registration is currently disabled."""


class DuplicateUserError(FactoryMindError):
    """Username or email already exists."""
