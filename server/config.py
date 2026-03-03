from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Always resolve .env relative to the project root (parent of server/)
# so `uv run uvicorn main:app` works from any CWD.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/factorymind.db"
    secret_key: str  # required – must be set in .env
    api_key_header: str = "X-API-Key"
    vllm_port: int = 8000
    hf_cache_dir: str = "/cache/huggingface"
    enable_registration: bool = True
    max_chat_history: int = 100

    # ── Model warm-up & pre-fetching ──────────────────────────────────────────
    # Model to load into GPU at server startup (eliminates cold-start latency).
    default_model: str = "llama-3.1-8b-awq"
    # When True, weights for ALL catalog models are downloaded to disk on
    # startup (no GPU use). This ensures instant model-swap without waiting
    # for a download the first time a user requests a different model.
    prefetch_all_models: bool = True
    # When True, passes --enforce-eager to vLLM, disabling torch.compile and
    # CUDA Graphs. Saves ~18-20 s per model swap at the cost of ~10-15%
    # steady-state throughput. Recommended for single-GPU hot-swap setups.
    fast_model_swap: bool = True

    @field_validator("secret_key")
    @classmethod
    def secret_key_min_length(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}


settings = Settings()
