"""LangGraph nodes for the ODIN pipeline.

Node inventory
--------------
generate_node     — initial code generation; injects past_run_context if present
execute_node      — sandboxed subprocess execution
analyze_error_node — six-category error classification
retrieve_node     — adaptive RAG over odin_knowledge collection
repair_node       — LLM repair with optional retrieved docs
verify_node       — check success / max-iterations gate
feedback_node     — re-generation driven by explicit user feedback (new)

The feedback_node is the entry point for the is_continuation=True path.
It does NOT trigger a run_memory lookup — that guarantee is enforced at the
graph level (feedback_node is only reachable when is_continuation=True, and
the run_memory query only happens outside the graph in backend.py before
graph invocation).
"""
from __future__ import annotations

import logging
from typing import Any

from odin.error_classifier import classify_error
from odin.executor import execute_code
from odin.llm import apply_feedback, generate_code, repair_code
from odin.rag.retriever import RAGRetriever
from odin.routing import should_retrieve
from odin.state import (
    CodeVersion,
    ErrorAnalysis,
    ExecutionRecord,
    GraphState,
    RetrievalRecord,
)

logger = logging.getLogger(__name__)

# RAGRetriever is instantiated once per process and reused.
# Previously it was re-created on every retrieve_node call, which reloaded the
# embedding model each time.  Moving to a module-level instance avoids that.
_retriever: RAGRetriever | None = None


def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_node(state: GraphState) -> dict[str, Any]:
    """Generate initial code from the requirement.

    Injects past_run_context into the prompt if the backend found a similar
    past run via run_memory before invoking the graph.
    """
    logger.info("=== GENERATE NODE ===")
    state["run_history"].add_stage_timestamp("generate")

    iteration = 1
    requirement = state["requirement"]
    past_ctx = state.get("past_run_context")

    is_multi_file = any(
        kw in requirement.lower()
        for kw in ["api", "server", "fastapi", "flask", "django", "module", "package"]
    )

    files, token_usage = generate_code(
        requirement=requirement,
        iteration=iteration,
        is_multi_file=is_multi_file,
        past_run_context=past_ctx,
    )

    token_usage.iteration = iteration
    state["run_history"].code_versions.append(CodeVersion(iteration=iteration, files=files))
    state["run_history"].token_usage.append(token_usage)

    if past_ctx:
        logger.info(
            "Generate used past-run context from run %s (similarity=%.2f)",
            past_ctx.run_id,
            past_ctx.similarity_score,
        )

    logger.info("Generated %d file(s) at iteration %d", len(files), iteration)
    return {"iteration": iteration, "current_files": files}


def execute_node(state: GraphState) -> dict[str, Any]:
    """Execute the current code in a sandboxed subprocess and verify tests."""
    logger.info("=== EXECUTE NODE ===")
    state["run_history"].add_stage_timestamp("execute")

    files = state["current_files"]
    iteration = state["iteration"]

    execution = execute_code(
        files,
        timeout=state["execution_timeout"],
        test_input=state["test_input"],
        test_cases=state.get("test_cases", ""),
        expected_output=state.get("expected_output", ""),
    )
    execution.iteration = iteration
    state["run_history"].executions.append(execution)

    logger.info(
        "Execution iter %d: %s (%.2fms, tests=%d/%d)",
        iteration,
        "SUCCESS" if execution.success else "FAILURE",
        execution.execution_time_ms,
        execution.tests_passed,
        execution.tests_total,
    )
    if not execution.success:
        logger.error("Error: %.200s", execution.stderr)

    return {"last_execution": execution, "success": execution.success}


def analyze_error_node(state: GraphState) -> dict[str, Any]:
    """Classify the execution error into one of the six categories."""
    logger.info("=== ANALYZE ERROR NODE ===")
    state["run_history"].add_stage_timestamp("analyze_error")

    execution = state["last_execution"]
    iteration = state["iteration"]

    if execution.success:
        logger.info("Execution successful — no error to analyse")
        return {"should_end": True}

    test_passed = bool(execution.test_passed) if execution.test_passed is not None else False
    category, reasoning = classify_error(execution, test_passed=test_passed)
    error_analysis = ErrorAnalysis(iteration=iteration, category=category, reasoning=reasoning)
    state["run_history"].error_analyses.append(error_analysis)

    logger.info("Error classified as '%s': %s", category, reasoning)
    return {"last_error_analysis": error_analysis}


def retrieve_node(state: GraphState) -> dict[str, Any]:
    """Conditionally retrieve from odin_knowledge based on experiment mode & error category."""
    logger.info("=== RETRIEVE NODE ===")
    state["run_history"].add_stage_timestamp("retrieve")

    error_analysis = state["last_error_analysis"]
    execution = state["last_execution"]
    iteration = state["iteration"]
    mode = state.get("experiment_mode", "adaptive_rag")

    if mode == "no_rag":
        do_retrieve = False
        decision_reason = "Retrieval disabled by experiment mode (no_rag / Config B)"
    elif mode == "always_rag":
        do_retrieve = True
        decision_reason = f"Retrieval forced by experiment mode (always_rag) for '{error_analysis.category}'"
    else:
        do_retrieve, decision_reason = should_retrieve(error_analysis.category)

    if not do_retrieve:
        logger.info("Skipping retrieval: %s", decision_reason)
        record = RetrievalRecord(
            iteration=iteration, triggered=False, decision_reason=decision_reason
        )
        state["run_history"].retrievals.append(record)
        return {"last_retrieval": record}

    logger.info("Triggering retrieval: %s", decision_reason)
    query = f"{error_analysis.category} error: {execution.stderr[:500]}"
    record = _get_retriever().retrieve(
        query=query, category=error_analysis.category, iteration=iteration
    )
    record.decision_reason = decision_reason
    state["run_history"].retrievals.append(record)

    logger.info("Retrieved %d chunks", len(record.chunks))
    return {"last_retrieval": record}


def repair_node(state: GraphState) -> dict[str, Any]:
    """Repair code given execution error, prior debugging history, and optional retrieved docs."""
    logger.info("=== REPAIR NODE ===")
    state["run_history"].add_stage_timestamp("repair")

    iteration = state["iteration"] + 1
    execution = state["last_execution"]
    retrieval = state["last_retrieval"]

    # Build concise history of earlier failed attempts (excluding current failure)
    run_history = state["run_history"]
    cat_by_iter = {ea.iteration: ea.category for ea in run_history.error_analyses}
    previous_attempts = [
        {
            "iteration": str(ex.iteration),
            "category": cat_by_iter.get(ex.iteration, "error"),
            "error": (ex.stderr or ex.traceback)[:250],
        }
        for ex in run_history.executions[:-1]
        if not ex.success
    ]

    files, token_usage = repair_code(
        requirement=state["requirement"],
        previous_files=state["current_files"],
        error_message=execution.stderr,
        traceback=execution.traceback,
        iteration=iteration,
        retrieval=retrieval if retrieval and retrieval.triggered else None,
        previous_attempts=previous_attempts or None,
    )

    token_usage.iteration = iteration
    state["run_history"].code_versions.append(CodeVersion(iteration=iteration, files=files))
    state["run_history"].token_usage.append(token_usage)

    logger.info("Repaired code at iteration %d", iteration)
    return {"iteration": iteration, "current_files": files}


def verify_node(state: GraphState) -> dict[str, Any]:
    """Check success / max-iterations gate and decide whether to continue."""
    logger.info("=== VERIFY NODE ===")
    state["run_history"].add_stage_timestamp("verify")

    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    success = state["success"]

    if success:
        logger.info("✓ Execution succeeded!")
        state["run_history"].finalize("success")
        return {"should_end": True}

    if iteration >= max_iterations:
        logger.warning("✗ Max iterations (%d) reached", max_iterations)
        state["run_history"].finalize("max_iterations_reached")
        return {"should_end": True}

    logger.info("Continuing → iteration %d / %d", iteration + 1, max_iterations)
    return {"should_end": False}


def feedback_node(state: GraphState) -> dict[str, Any]:
    """Re-generate code based on explicit user feedback.

    Entry point for the is_continuation=True path.  The user told us the
    current code isn't acceptable and provided a correction note.

    Key invariant: this node is ONLY reachable when is_continuation=True.
    The run_memory query in backend.py is skipped for continuations, so
    past_run_context is None here.  That is correct and expected.

    After re-generating, control flows to execute_node so the new code is
    validated the same way as any other iteration.
    """
    logger.info("=== FEEDBACK NODE ===")
    state["run_history"].add_stage_timestamp("feedback")

    # Increment iteration so this attempt gets its own iteration_N folder
    iteration = state["iteration"] + 1
    user_feedback = state.get("user_feedback", "")

    if not user_feedback:
        logger.warning("feedback_node reached with empty user_feedback — treating as repair retry")

    files, token_usage = apply_feedback(
        requirement=state["requirement"],
        current_files=state["current_files"],
        user_feedback=user_feedback,
        iteration=iteration,
    )

    token_usage.iteration = iteration
    state["run_history"].code_versions.append(CodeVersion(iteration=iteration, files=files))
    state["run_history"].token_usage.append(token_usage)

    logger.info("Feedback revision produced %d file(s) at iteration %d", len(files), iteration)
    # Reset is_continuation so subsequent loops don't re-enter feedback_node
    return {
        "iteration": iteration,
        "current_files": files,
        "is_continuation": False,
        "user_feedback": "",
    }
