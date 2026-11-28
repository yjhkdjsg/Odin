"""State management for the ODIN LangGraph workflow."""
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


ErrorCategory = Literal[
    "syntax",
    "runtime",
    "logical",
    "library_api",
    "version_specific",
    "uncommon_pattern",
]

ExperimentMode = Literal[
    "direct",        # Config A: one-shot generation, no repair
    "no_rag",        # Config B: iterative self-debugging without RAG
    "adaptive_rag",  # Config C: iterative self-debugging with selective RAG
    "always_rag",    # Config D: iterative self-debugging with always-on RAG
]


# ---------------------------------------------------------------------------
# Sub-models (unchanged schema, kept identical for back-compat)
# ---------------------------------------------------------------------------

class FileContent(BaseModel):
    path: str
    content: str


class ExecutionRecord(BaseModel):
    iteration: int
    stdout: str
    stderr: str
    exit_code: int
    traceback: str
    success: bool
    execution_time_ms: float
    tests_passed: int = 0
    tests_total: int = 0
    test_passed: bool | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorAnalysis(BaseModel):
    iteration: int
    category: ErrorCategory
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class RetrievalRecord(BaseModel):
    iteration: int
    triggered: bool
    decision_reason: str
    query: str = ""
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    total_tokens: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class CodeVersion(BaseModel):
    iteration: int
    files: list[FileContent]
    timestamp: datetime = Field(default_factory=datetime.now)


class TokenUsage(BaseModel):
    iteration: int
    call_type: Literal["generate", "repair", "feedback"]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Past-run context  (produced by run_memory lookup, consumed by generate_node)
# ---------------------------------------------------------------------------

class PastRunContext(BaseModel):
    """A similar past run found via run_memory semantic search.

    Carries both the embedding-level summary and the actual file content read
    from disk so generate_node can include it verbatim in the prompt.
    """
    run_id: str
    requirement: str
    outcome: Literal["success", "failure", "max_iterations_reached"]
    similarity_score: float
    run_folder: str                            # absolute path on disk
    error_categories_seen: list[str]           # e.g. ["runtime", "library_api"]
    # Actual source files from the final iteration folder (two-step read)
    final_files: list[dict[str, str]] = Field(default_factory=list)
    # Embedded summary text (for reference / logging)
    summary_text: str = ""


# ---------------------------------------------------------------------------
# RunHistory
# ---------------------------------------------------------------------------

class RunHistory(BaseModel):
    """Complete history of a single pipeline run."""

    run_id: UUID = Field(default_factory=uuid4)
    requirement: str
    experiment_mode: str = "adaptive_rag"
    test_input: str = ""
    test_cases: str = ""
    expected_output: str = ""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None

    code_versions: list[CodeVersion] = Field(default_factory=list)
    executions: list[ExecutionRecord] = Field(default_factory=list)
    error_analyses: list[ErrorAnalysis] = Field(default_factory=list)
    retrievals: list[RetrievalRecord] = Field(default_factory=list)
    token_usage: list[TokenUsage] = Field(default_factory=list)

    stage_timestamps: dict[str, list[datetime]] = Field(default_factory=dict)

    final_status: Literal["success", "failure", "max_iterations_reached"] | None = None
    error_message: str | None = None
    total_iterations: int = 0

    def add_stage_timestamp(self, stage_name: str, timestamp: datetime | None = None) -> None:
        if stage_name not in self.stage_timestamps:
            self.stage_timestamps[stage_name] = []
        self.stage_timestamps[stage_name].append(timestamp or datetime.now())

    def finalize(self, status: Literal["success", "failure", "max_iterations_reached"]) -> None:
        self.end_time = datetime.now()
        self.final_status = status
        self.total_iterations = len(self.code_versions)


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """State passed through the LangGraph workflow nodes."""

    # ── Input ────────────────────────────────────────────────────────────
    requirement: str
    test_input: str                     # piped to stdin during execution
    test_cases: str                     # optional Python assert test suite
    expected_output: str                # optional expected stdout for DSA checks
    experiment_mode: ExperimentMode     # direct | no_rag | adaptive_rag | always_rag

    # ── Re-prompt / continuation fields ─────────────────────────────────
    # is_continuation=True  → user gave feedback on an active unresolved run.
    #   The graph skips run_memory lookup and jumps straight to feedback_node.
    # is_continuation=False → new request; run_memory is queried first.
    is_continuation: bool
    user_feedback: str                  # non-empty only when is_continuation=True

    # ── Past-run memory context (new request only) ───────────────────────
    # Populated by the backend before graph invocation when a sufficiently
    # similar past run is found.  None if no match or is_continuation=True.
    past_run_context: PastRunContext | None

    # ── Pipeline state ───────────────────────────────────────────────────
    iteration: int
    # NOTE: NOT annotated with operator.add.  Each node that sets current_files
    # replaces the list outright.  The previous operator.add annotation caused
    # LangGraph to concatenate stale file lists across iterations.
    current_files: list[FileContent]
    last_execution: ExecutionRecord | None
    last_error_analysis: ErrorAnalysis | None
    last_retrieval: RetrievalRecord | None

    # ── Configuration ────────────────────────────────────────────────────
    max_iterations: int
    execution_timeout: int

    # ── Run history (mutated in-place by every node) ─────────────────────
    run_history: RunHistory

    # ── Control flags ────────────────────────────────────────────────────
    success: bool
    should_end: bool
