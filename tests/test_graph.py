import pytest
from unittest.mock import patch, MagicMock

import odin.nodes as odin_nodes
from odin.graph import build_graph
from odin.state import (
    ExecutionRecord,
    GraphState,
    RetrievalRecord,
    RunHistory,
    TokenUsage,
)


def _make_token_usage(call_type: str = "generate", iteration: int = 1) -> TokenUsage:
    return TokenUsage(
        iteration=iteration,
        call_type=call_type,  # type: ignore[arg-type]
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )


def _make_exec(success: bool, stderr: str = "", traceback: str = "") -> ExecutionRecord:
    return ExecutionRecord(
        iteration=1,
        stdout="ok" if success else "",
        stderr=stderr,
        exit_code=0 if success else 1,
        traceback=traceback,
        success=success,
        execution_time_ms=5.0,
    )


def _make_state(
    run_history: RunHistory,
    max_iterations: int = 5,
    experiment_mode: str = "adaptive_rag",
) -> GraphState:
    return {
        "requirement": run_history.requirement,
        "test_input": "",
        "test_cases": "",
        "expected_output": "",
        "experiment_mode": experiment_mode,  # type: ignore[typeddict-item]
        "is_continuation": False,
        "user_feedback": "",
        "past_run_context": None,
        "iteration": 0,
        "current_files": [],
        "last_execution": None,
        "last_error_analysis": None,
        "last_retrieval": None,
        "max_iterations": max_iterations,
        "execution_timeout": 5,
        "run_history": run_history,
        "success": False,
        "should_end": False,
    }


@pytest.fixture
def mock_graph_deps():
    odin_nodes._retriever = None
    with patch("odin.nodes.generate_code") as mock_gen, \
         patch("odin.nodes.repair_code") as mock_rep, \
         patch("odin.nodes.execute_code") as mock_exec, \
         patch("odin.nodes.classify_error") as mock_class:
        yield mock_gen, mock_rep, mock_exec, mock_class
    odin_nodes._retriever = None


@pytest.mark.asyncio
async def test_graph_first_attempt_success(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps
    mock_gen.return_value = ([], _make_token_usage("generate", 1))
    mock_exec.return_value = _make_exec(success=True)

    run_history = RunHistory(requirement="test")
    graph = build_graph()
    final_state = await graph.ainvoke(_make_state(run_history, max_iterations=3))

    assert final_state["success"] is True
    assert final_state["should_end"] is True
    assert final_state["iteration"] == 1
    assert run_history.final_status == "success"
    assert mock_gen.call_count == 1
    assert mock_exec.call_count == 1
    assert mock_class.call_count == 0
    assert mock_rep.call_count == 0


@pytest.mark.asyncio
async def test_graph_direct_generation_config_a(mock_graph_deps):
    """max_iterations=1 executes once and stops without repairing on failure."""
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps
    mock_gen.return_value = ([], _make_token_usage("generate", 1))
    mock_exec.return_value = _make_exec(success=False, stderr="error1", traceback="error1")

    run_history = RunHistory(requirement="test")
    graph = build_graph()
    final_state = await graph.ainvoke(_make_state(run_history, max_iterations=1))

    assert final_state["success"] is False
    assert final_state["should_end"] is True
    assert final_state["iteration"] == 1
    assert run_history.final_status == "max_iterations_reached"
    assert mock_gen.call_count == 1
    assert mock_exec.call_count == 1
    assert mock_rep.call_count == 0


@pytest.mark.asyncio
async def test_graph_sequence_fail_then_pass(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps

    mock_gen.return_value = ([], _make_token_usage("generate", 1))
    mock_exec.side_effect = [
        _make_exec(success=False, stderr="error1", traceback="error1"),
        _make_exec(success=True),
    ]
    mock_class.return_value = ("logical", "Test assertion failed")
    mock_rep.return_value = ([], _make_token_usage("repair", 2))

    run_history = RunHistory(requirement="test")
    graph = build_graph()
    final_state = await graph.ainvoke(_make_state(run_history, max_iterations=5))

    assert final_state["success"] is True
    assert final_state["should_end"] is True
    assert final_state["iteration"] == 2
    assert run_history.final_status == "success"
    assert mock_gen.call_count == 1
    assert mock_exec.call_count == 2
    assert mock_class.call_count == 1
    assert mock_rep.call_count == 1


@pytest.mark.asyncio
async def test_graph_max_iterations(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps

    mock_gen.return_value = ([], _make_token_usage("generate", 1))
    mock_rep.side_effect = [
        ([], _make_token_usage("repair", 2)),
        ([], _make_token_usage("repair", 3)),
    ]
    mock_class.return_value = ("logical", "Test assertion failed")
    mock_exec.side_effect = [
        _make_exec(success=False, stderr="error1", traceback="error1"),
        _make_exec(success=False, stderr="error2", traceback="error2"),
        _make_exec(success=False, stderr="error3", traceback="error3"),
    ]

    run_history = RunHistory(requirement="test")
    graph = build_graph()
    final_state = await graph.ainvoke(_make_state(run_history, max_iterations=3))

    assert final_state["success"] is False
    assert final_state["should_end"] is True
    assert run_history.final_status == "max_iterations_reached"
    assert final_state["iteration"] == 3
    # Every iteration (initial + 2 repairs) must actually be executed
    assert mock_gen.call_count == 1
    assert mock_rep.call_count == 2
    assert mock_exec.call_count == 3


@pytest.mark.asyncio
async def test_retrieval_routing_reevaluated(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps

    mock_gen.return_value = ([], _make_token_usage("generate", 1))
    mock_rep.side_effect = [
        ([], _make_token_usage("repair", 2)),
        ([], _make_token_usage("repair", 3)),
    ]

    mock_exec.side_effect = [
        _make_exec(success=False, stderr="SyntaxError", traceback="SyntaxError"),
        _make_exec(success=False, stderr="AttributeError", traceback="AttributeError"),
        _make_exec(success=True),
    ]

    # First time syntax (skip retrieval), second time library_api (trigger retrieval)
    mock_class.side_effect = [
        ("syntax", "Parser failure: SyntaxError"),
        ("library_api", "AttributeError in pandas context"),
    ]

    run_history = RunHistory(requirement="test")
    graph = build_graph()

    with patch("odin.nodes.RAGRetriever") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = RetrievalRecord(
            iteration=2,
            triggered=True,
            decision_reason="Category 'library_api' triggers retrieval per routing table",
            query="library_api error: AttributeError",
            chunks=[{"content": "Use pd.concat", "metadata": {}, "similarity_score": 0.9}],
            retrieval_latency_ms=5.0,
            total_tokens=10,
        )
        mock_rag_cls.return_value = mock_rag

        final_state = await graph.ainvoke(_make_state(run_history, max_iterations=5))

        assert final_state["success"] is True
        assert run_history.final_status == "success"
        rets = run_history.retrievals
        assert len(rets) == 2
        assert rets[0].triggered is False  # syntax
        assert rets[1].triggered is True   # library_api


@pytest.mark.asyncio
async def test_graph_experiment_modes_no_rag_and_always_rag(mock_graph_deps):
    mock_gen, mock_rep, mock_exec, mock_class = mock_graph_deps
    graph = build_graph()

    # 1. Config B (no_rag): must SKIP retrieval even for library_api
    mock_gen.return_value = ([], _make_token_usage("generate", 1))
    mock_rep.return_value = ([], _make_token_usage("repair", 2))
    mock_exec.side_effect = [
        _make_exec(success=False, stderr="AttributeError", traceback="AttributeError"),
        _make_exec(success=True),
    ]
    mock_class.return_value = ("library_api", "pandas API misuse")

    rh_no_rag = RunHistory(requirement="test no_rag")
    await graph.ainvoke(_make_state(rh_no_rag, max_iterations=3, experiment_mode="no_rag"))
    assert len(rh_no_rag.retrievals) == 1
    assert rh_no_rag.retrievals[0].triggered is False

    # 2. Config D (always_rag): must TRIGGER retrieval even for syntax
    mock_exec.side_effect = [
        _make_exec(success=False, stderr="SyntaxError", traceback="SyntaxError"),
        _make_exec(success=True),
    ]
    mock_class.return_value = ("syntax", "SyntaxError")

    rh_always = RunHistory(requirement="test always_rag")
    with patch("odin.nodes.RAGRetriever") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = RetrievalRecord(
            iteration=1,
            triggered=True,
            decision_reason="forced",
            query="syntax error",
            chunks=[],
            retrieval_latency_ms=2.0,
            total_tokens=0,
        )
        mock_rag_cls.return_value = mock_rag
        await graph.ainvoke(_make_state(rh_always, max_iterations=3, experiment_mode="always_rag"))

    assert len(rh_always.retrievals) == 1
    assert rh_always.retrievals[0].triggered is True

