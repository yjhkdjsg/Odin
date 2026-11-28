"""Unit tests for the FastAPI backend and the research experiment analysis pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

import odin.api as odin_api
from experiments.analyze_results import analyze_dataframe
from experiments.run_experiments import load_custom_benchmark
from odin.state import CodeVersion, ExecutionRecord, FileContent, RunHistory


def _make_sample_history() -> RunHistory:
    rh = RunHistory(requirement="add two numbers", experiment_mode="adaptive_rag")
    rh.code_versions.append(
        CodeVersion(iteration=1, files=[FileContent(path="main.py", content="print(3)")])
    )
    rh.executions.append(
        ExecutionRecord(
            iteration=1, stdout="3\n", stderr="", exit_code=0,
            traceback="", success=True, execution_time_ms=10.0,
            tests_passed=1, tests_total=1, test_passed=True,
        )
    )
    rh.finalize("success")
    return rh


def test_fastapi_health_and_run_endpoints(tmp_path: Path):
    client = TestClient(odin_api.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    sample_rh = _make_sample_history()
    mock_backend = MagicMock()
    mock_backend.run_pipeline = AsyncMock(return_value=sample_rh)
    mock_backend.continue_run = AsyncMock(return_value=sample_rh)
    mock_backend.get_run_history.return_value = sample_rh
    mock_backend.compute_metrics.return_value = {
        "run_id": str(sample_rh.run_id),
        "success": True,
        "total_iterations": 1,
        "total_llm_tokens": 100,
    }

    with patch("odin.api.get_backend", return_value=mock_backend):
        resp = client.post(
            "/runs",
            json={
                "requirement": "add two numbers",
                "expected_output": "3",
                "experiment_mode": "adaptive_rag",
                "output_dir": str(tmp_path),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_history"]["final_status"] == "success"
        assert body["saved_output_dir"] is not None
        assert (Path(body["saved_output_dir"]) / "main.py").exists()

        fb_resp = client.post(
            f"/runs/{sample_rh.run_id}/feedback",
            json={"user_feedback": "also print 4", "test_input": "1 2"},
        )
        assert fb_resp.status_code == 200
        mock_backend.continue_run.assert_awaited_once()


def test_custom_benchmark_covers_all_six_categories():
    tasks = load_custom_benchmark()
    assert len(tasks) >= 12
    categories = {t["target_category"] for t in tasks}
    expected = {
        "syntax",
        "runtime",
        "logical",
        "library_api",
        "version_specific",
        "uncommon_pattern",
    }
    assert expected.issubset(categories)


def test_analyze_results_generates_csvs_and_plots(tmp_path: Path):
    df = pd.DataFrame([
        {
            "config": "config_a_direct", "task_id": "t1", "target_category": "syntax",
            "success": True, "test_pass_rate": 1.0, "first_attempt_success": True,
            "partial_repair": False, "total_iterations": 1, "wall_time_seconds": 1.2,
            "total_tool_calls": 1, "retrieval_triggered_count": 0, "retrieval_decisions": 0,
            "total_input_tokens": 100, "total_output_tokens": 50, "total_llm_tokens": 150,
            "retrieved_context_tokens": 0, "retrieval_latency_ms": 0.0,
            "retrieval_precision": None, "retrieval_recall": None,
        },
        {
            "config": "config_b_no_rag", "task_id": "t2", "target_category": "version_specific",
            "success": False, "test_pass_rate": 0.5, "first_attempt_success": False,
            "partial_repair": True, "total_iterations": 3, "wall_time_seconds": 4.5,
            "total_tool_calls": 3, "retrieval_triggered_count": 0, "retrieval_decisions": 2,
            "total_input_tokens": 400, "total_output_tokens": 180, "total_llm_tokens": 580,
            "retrieved_context_tokens": 0, "retrieval_latency_ms": 0.0,
            "retrieval_precision": None, "retrieval_recall": None,
        },
        {
            "config": "config_c_adaptive_rag", "task_id": "t2", "target_category": "version_specific",
            "success": True, "test_pass_rate": 1.0, "first_attempt_success": False,
            "partial_repair": False, "total_iterations": 2, "wall_time_seconds": 3.1,
            "total_tool_calls": 3, "retrieval_triggered_count": 1, "retrieval_decisions": 1,
            "total_input_tokens": 350, "total_output_tokens": 120, "total_llm_tokens": 470,
            "retrieved_context_tokens": 80, "retrieval_latency_ms": 18.0,
            "retrieval_precision": 1.0, "retrieval_recall": 0.67,
        },
    ])

    results = analyze_dataframe(df, out_dir=tmp_path)
    assert len(results["summary_by_config"]) == 3
    assert (tmp_path / "summary_by_config.csv").exists()
    assert (tmp_path / "summary_by_category.csv").exists()
    assert (tmp_path / "negative_effects.csv").exists()
    assert (tmp_path / "success_rate_by_config.png").exists()
    assert (tmp_path / "tokens_and_latency_by_config.png").exists()
    assert (tmp_path / "category_comparison.png").exists()
