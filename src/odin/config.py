"""Configuration management for ODIN.

All paths are relative to the repository root (the working directory when the
user runs `python -m odin` from inside the cloned repo).

Folder layout on disk:
    outputs/                        ← run folders (one per run)
    chroma_data/odin_knowledge/     ← static docs/API-ref knowledge base
    chroma_data/run_memory/         ← past-run memory (second collection)
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class OdinConfig(BaseModel):
    """Global configuration for ODIN framework."""

    # ── LLM ───────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    )

    # ── Pipeline ──────────────────────────────────────────────────────────
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("ODIN_MAX_ITERATIONS", "5"))
    )
    execution_timeout: int = Field(
        default_factory=lambda: int(os.getenv("ODIN_EXECUTION_TIMEOUT", "10"))
    )
    sandbox_memory_limit_mb: int = Field(
        default_factory=lambda: int(os.getenv("ODIN_SANDBOX_MEMORY_MB", "1024"))
    )
    sandbox_file_size_limit_mb: int = Field(
        default_factory=lambda: int(os.getenv("ODIN_SANDBOX_FILE_SIZE_MB", "10"))
    )
    sandbox_mode: str = Field(
        default_factory=lambda: os.getenv("ODIN_SANDBOX_MODE", "subprocess").lower()
    )
    sandbox_docker_image: str = Field(
        default_factory=lambda: os.getenv("ODIN_SANDBOX_DOCKER_IMAGE", "python:3.12-slim")
    )
    repair_token_limit: int = Field(
        default_factory=lambda: int(os.getenv("ODIN_REPAIR_TOKEN_LIMIT", "30000"))
    )

    # ── Outputs (run folders + per-iteration files) ────────────────────────
    outputs_path: Path = Field(
        default_factory=lambda: Path(os.getenv("ODIN_OUTPUTS_PATH", "./outputs"))
    )

    # ── ChromaDB (single PersistentClient directory, two collections inside)
    chroma_data_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("ODIN_CHROMA_DATA_PATH", "./chroma_data")
        )
    )

    # Collection names — kept in config so tests can override them easily
    knowledge_collection_name: str = "odin_knowledge"
    run_memory_collection_name: str = "run_memory"

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3

    # Minimum cosine similarity (0–1) for a past run to be surfaced.
    # ChromaDB returns L2 distances; we convert via  sim = 1 - dist.
    # 0.70 means "meaningfully similar"; lower → more noise, higher → fewer hits.
    run_memory_similarity_threshold: float = Field(
        default_factory=lambda: float(
            os.getenv("ODIN_MEMORY_SIMILARITY_THRESHOLD", "0.70")
        )
    )

    def validate_config(self) -> None:
        """Raise if required configuration is missing."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

    def ensure_directories(self) -> None:
        """Create all required directories if they do not already exist."""
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        self.chroma_data_path.mkdir(parents=True, exist_ok=True)

    # ── Back-compat shims used by existing code that imported the old names ─
    @property
    def rag_db_path(self) -> Path:
        """Legacy alias → chroma_data_path."""
        return self.chroma_data_path

    @property
    def run_history_path(self) -> Path:
        """Legacy alias → outputs_path."""
        return self.outputs_path


# Global singleton — imported everywhere
config = OdinConfig()
