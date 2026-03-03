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


# ── Asset domain exceptions ────────────────────────────────────────────────────


class EnvironmentNotFoundError(FactoryMindError):
    """Environment does not exist."""


class AssetNotFoundError(FactoryMindError):
    """Asset does not exist."""


class ConnectorNotFoundError(FactoryMindError):
    """Connector configuration does not exist."""


class EventNotFoundError(FactoryMindError):
    """Health event does not exist."""


class AccessDeniedError(FactoryMindError):
    """The requesting user does not own the requested resource."""
