"""FastAPI backend server for ODIN.

Provides a clean REST API over OdinBackend so web frontends (Streamlit or others)
and external experiment scripts can run pipelines, send feedback, inspect run
history, and retrieve evaluation metrics.

Run with:
    python -m odin.api
or:
    uvicorn odin.api:app --reload --port 8000
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from odin.backend import OdinBackend
from odin.config import config
from odin.persistence import RunHistoryPersistence, _make_run_dir_name
from odin.state import ExperimentMode, RunHistory

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ODIN API",
    description="Agentic Self-Debugging Framework for Automated Code Repair using Adaptive RAG",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_backend: OdinBackend | None = None


def get_backend() -> OdinBackend:
    """Lazily instantiate a shared OdinBackend instance."""
    global _backend
    if _backend is None:
        _backend = OdinBackend()
    return _backend


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CreateRunRequest(BaseModel):
    requirement: str = Field(..., min_length=1)
    test_input: str = ""
    test_cases: str = ""
    expected_output: str = ""
    experiment_mode: ExperimentMode = "adaptive_rag"
    max_iterations: Optional[int] = Field(default=None, ge=1, le=10)
    execution_timeout: Optional[int] = Field(default=None, ge=1, le=60)
    use_run_memory: Optional[bool] = None
    output_dir: Optional[str] = None


class FeedbackRequest(BaseModel):
    user_feedback: str = Field(..., min_length=1)
    test_input: Optional[str] = None
    test_cases: Optional[str] = None
    expected_output: Optional[str] = None
    execution_timeout: Optional[int] = Field(default=None, ge=1, le=60)
    output_dir: Optional[str] = None


def _copy_if_requested(run_history: RunHistory, output_dir: Optional[str]) -> Optional[str]:
    if not output_dir or run_history.final_status != "success" or not run_history.code_versions:
        return None
    base = Path(output_dir).expanduser().resolve()
    if base.exists() and base.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Output path '{base}' is a file, not a directory.",
        )
    dest_dir = base / _make_run_dir_name(run_history.run_id, run_history.requirement)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in run_history.code_versions[-1].files:
        target = dest_dir / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")
    return str(dest_dir)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return service health, active model, sandbox mode, and saved run count."""
    persistence = RunHistoryPersistence()
    return {
        "status": "ok",
        "model": config.gemini_model,
        "sandbox_mode": config.sandbox_mode,
        "max_iterations": config.max_iterations,
        "execution_timeout": config.execution_timeout,
        "saved_runs": len(persistence.list_runs()),
    }


@app.post("/runs")
async def create_run(req: CreateRunRequest) -> dict[str, Any]:
    """Run a new ODIN self-debugging pipeline."""
    backend = get_backend()
    run_history = await backend.run_pipeline(
        requirement=req.requirement,
        test_input=req.test_input,
        test_cases=req.test_cases,
        expected_output=req.expected_output,
        experiment_mode=req.experiment_mode,
        max_iterations=req.max_iterations,
        execution_timeout=req.execution_timeout,
        use_run_memory=req.use_run_memory,
    )
    saved_dir = _copy_if_requested(run_history, req.output_dir)
    metrics = backend.compute_metrics(str(run_history.run_id))
    return {
        "run_history": run_history.model_dump(mode="json"),
        "metrics": metrics,
        "saved_output_dir": saved_dir,
    }


@app.post("/runs/{run_id}/feedback")
async def send_feedback(run_id: str, req: FeedbackRequest) -> dict[str, Any]:
    """Continue an existing run with explicit user feedback."""
    backend = get_backend()
    try:
        run_history = backend.get_run_history(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    updated = await backend.continue_run(
        run_history=run_history,
        user_feedback=req.user_feedback,
        test_input=req.test_input,
        test_cases=req.test_cases,
        expected_output=req.expected_output,
        execution_timeout=req.execution_timeout,
    )
    saved_dir = _copy_if_requested(updated, req.output_dir)
    metrics = backend.compute_metrics(str(updated.run_id))
    return {
        "run_history": updated.model_dump(mode="json"),
        "metrics": metrics,
        "saved_output_dir": saved_dir,
    }


@app.get("/runs")
def list_runs() -> dict[str, Any]:
    """List all persisted runs, newest first."""
    persistence = RunHistoryPersistence()
    return {"runs": persistence.list_runs()}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Fetch full run history and metrics for a specific run ID."""
    persistence = RunHistoryPersistence()
    try:
        run_history = persistence.load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "run_history": run_history.model_dump(mode="json"),
        "metrics": persistence.compute_metrics(run_id),
    }


def main() -> None:
    """Start the FastAPI server using Uvicorn."""
    import uvicorn
    uvicorn.run("odin.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
