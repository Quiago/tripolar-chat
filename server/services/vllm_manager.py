"""vLLM process manager – loads exactly one model at a time into GPU memory."""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)


def _read_hf_token() -> Optional[str]:
    """Return the HuggingFace token from env vars or the token file.

    Reads from the ORIGINAL HF_HOME (before we override it for the weight
    cache) so that `hf auth login` tokens are always found.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(var, "").strip()
        if token:
            return token

    # Resolve token file from current HF_HOME (set before our override)
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    token_file = Path(hf_home) / "token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token

    return None


def _extract_error(stderr_text: str) -> str:
    """Return a single-line error summary from vLLM's stderr.

    Looks for the last line that starts with a known exception class (e.g.
    ``OSError:``, ``RepositoryNotFoundError:``), which is the actual root
    cause.  Falls back to the last non-empty, non-help-text line.
    """
    _worker_prefix = re.compile(r"^\s*\([A-Za-z]+ pid=\d+\)\s*")
    _exc_re = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:Error|Exception|Warning):")

    candidates: list[str] = []
    for raw in stderr_text.splitlines():
        line = _worker_prefix.sub("", raw).strip()
        if line and not line.startswith("File ") and not line.startswith("During handling"):
            candidates.append(line)

    # Prefer the last line that looks like an exception class
    for line in reversed(candidates):
        if _exc_re.match(line):
            return line[:200]

    return (candidates[-1][:200] if candidates else "unknown error – see server logs")


@dataclass
class ModelConfig:
    repo: str
    quantization: Optional[str]
    tensor_parallel: int
    gpu_memory_util: float
    max_model_len: int
    description: str


MODEL_CATALOG: Dict[str, ModelConfig] = {
    "llama-3.1-8b-instruct": ModelConfig(
        # REPO OFICIAL: Pesos originales de Meta (BF16).
        # Requiere: ~16GB de VRAM y que pongas tu Token de Hugging Face.
        repo="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        quantization=None,
        tensor_parallel=1,
        gpu_memory_util=0.85,
        max_model_len=8192,
        description="Meta Llama 3.1 8B Instruct (Original - Pesado)"
    ),
    
    "llama-3.1-8b-awq": ModelConfig(
        # REPO DE LA COMUNIDAD: Cuantización AWQ oficial de Hugging Face (hugging-quants).
        # Requiere: Solo ~6.5GB de VRAM. ¡NO requiere Token de HF porque es público!
        repo="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 
        quantization="awq",
        tensor_parallel=1,
        gpu_memory_util=0.90,
        max_model_len=8192,
        description="Llama 3.1 8B 4-bit AWQ (Eficiente en VRAM)"
    ),
    "mistral-7b-awq": ModelConfig(
        repo="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        quantization="awq",
        tensor_parallel=1,
        gpu_memory_util=0.90,
        max_model_len=8192,
        description="Mistral 7B – good reasoning, compact",
    ),
    "qwen-2.5-7b": ModelConfig(
        repo="Qwen/Qwen2.5-7B-Instruct",
        quantization=None,
        tensor_parallel=1,
        gpu_memory_util=0.85,
        max_model_len=8192,
        description="Qwen 2.5 7B – multilingual",
    ),
}


class VLLMModelManager:
    """Manages a single vLLM process with hot-swapping between models."""

    def __init__(self, port: int, cache_dir: str) -> None:
        self.port = port
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_model: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None

    # ── status ────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if vLLM is healthy and accepting requests."""
        try:
            r = httpx.get(f"http://localhost:{self.port}/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def is_cached(self, model_name: str) -> bool:
        """Return True if the model's weights exist in the local HF cache."""
        config = MODEL_CATALOG.get(model_name)
        if not config:
            return False
        # HF Hub stores models under hub/models--<org>--<name>/
        slug = "models--" + config.repo.replace("/", "--")
        candidate = self.cache_dir / "hub" / slug
        return candidate.exists() and any(candidate.iterdir())

    def get_info(self) -> dict:
        """Snapshot suitable for embedding in /health responses."""
        return {
            "current_model": self.current_model,
            "is_ready": self.is_ready(),
            "catalog": list(MODEL_CATALOG.keys()),
        }

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Terminate the running vLLM process and free GPU memory."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            time.sleep(3)  # allow GPU memory to be released
        self.current_model = None
        self.process = None

    def start(self, model_name: str) -> None:
        """Start vLLM with *model_name*. Blocks until healthy or raises."""
        if model_name not in MODEL_CATALOG:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_CATALOG)}"
            )

        config = MODEL_CATALOG[model_name]

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.repo,
            "--port", str(self.port),
            "--host", "0.0.0.0",
            "--gpu-memory-utilization", str(config.gpu_memory_util),
            "--max-model-len", str(config.max_model_len),
            "--dtype", "auto",
            "--enable-prefix-caching",
        ]
        if config.quantization:
            cmd.extend(["--quantization", config.quantization])

        # Read the HF token BEFORE overriding HF_HOME so we always find
        # the token saved by `hf auth login` (stored in the original HF_HOME).
        hf_token = _read_hf_token()

        env = os.environ.copy()
        env["HF_HOME"] = str(self.cache_dir)
        env["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Pass the token explicitly so vLLM can authenticate even though
        # HF_HOME now points to our custom cache directory.
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token  # legacy env var

        self.process = subprocess.Popen(
            cmd,
            env=env,
            preexec_fn=os.setsid,
            stdout=None,           # flow directly to server terminal
            stderr=subprocess.PIPE,
        )

        deadline = time.time() + 300  # 5-minute total timeout
        while time.time() < deadline:
            if self.is_ready():
                self.current_model = model_name
                return
            if self.process.poll() is not None:
                _, stderr = self.process.communicate()
                log.error("vLLM exited. Full stderr:\n%s", stderr.decode())
                raise RuntimeError(f"vLLM exited: {_extract_error(stderr.decode())}")
            time.sleep(2)

        self.stop()
        raise TimeoutError("vLLM did not become ready within 300 s")

    def ensure_loaded(self, model_name: str) -> None:
        """Load *model_name*; swap if a different model is running."""
        if self.current_model == model_name and self.is_ready():
            return
        if self.current_model:
            self.stop()
        self.start(model_name)

    # ── inference ─────────────────────────────────────────────────────────────

    async def aforward_chat(
        self,
        messages: List[dict],
        model_name: str,
        **kwargs,
    ) -> dict:
        """Non-streaming chat request forwarded to the vLLM process."""
        if not self.is_ready():
            raise RuntimeError("vLLM is not ready")
        payload = {
            "model": MODEL_CATALOG[model_name].repo,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json=payload,
                timeout=300.0,
            )
            r.raise_for_status()
            return r.json()


# ── module-level singleton ────────────────────────────────────────────────────

_instance: Optional[VLLMModelManager] = None


def get_vllm_manager() -> VLLMModelManager:
    """Return (and lazily create) the global VLLMModelManager singleton."""
    global _instance
    if _instance is None:
        from server.config import settings

        _instance = VLLMModelManager(
            port=settings.vllm_port,
            cache_dir=settings.hf_cache_dir,
        )
    return _instance
