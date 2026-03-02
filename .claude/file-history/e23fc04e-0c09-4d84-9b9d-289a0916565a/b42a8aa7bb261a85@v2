from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/factorymind.db"
    secret_key: str  # required – must be set in .env
    api_key_header: str = "X-API-Key"
    vllm_port: int = 8000
    hf_cache_dir: str = "/cache/huggingface"
    enable_registration: bool = True
    max_chat_history: int = 100

    @field_validator("secret_key")
    @classmethod
    def secret_key_min_length(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
