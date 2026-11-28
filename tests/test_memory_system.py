"""Unit tests for the folder-based memory system.

Covers:
1. A new run correctly creates the iteration folder structure.
2. A completed run gets correctly embedded into run_memory.
3. A new request similar to a past run retrieves that run and reads its files.
4. A same-run re-prompt (is_continuation=True) does NOT trigger run_memory.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from odin.persistence import RunHistoryPersistence, _make_run_dir_name
from odin.state import (
    CodeVersion,
    ErrorAnalysis,
    ExecutionRecord,
    FileContent,
    PastRunContext,
    RetrievalRecord,
    RunHistory,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_outputs(tmp_path: Path) -> Path:
    """A temporary outputs/ directory."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return outputs


@pytest.fixture
def sample_run(tmp_outputs: Path) -> RunHistory:
    """A minimal completed RunHistory with two iterations."""
    rh = RunHistory(requirement="write a palindrome checker")

    rh.code_versions.append(CodeVersion(
        iteration=1,
        files=[FileContent(path="main.py", content="def is_palindrome(s): return s == s  # bug")],
    ))
    rh.executions.append(ExecutionRecord(
        iteration=1, stdout="", stderr="AssertionError", exit_code=1,
        traceback="AssertionError", success=False, execution_time_ms=12.0,
    ))
    rh.error_analyses.append(ErrorAnalysis(
        iteration=1, category="logical", reasoning="Test assertion failed",
    ))
    rh.retrievals.append(RetrievalRecord(
        iteration=1, triggered=False, decision_reason="logical skips retrieval",
    ))
    rh.token_usage.append(TokenUsage(
        iteration=1, call_type="generate",
        input_tokens=100, output_tokens=50, total_tokens=150,
    ))

    rh.code_versions.append(CodeVersion(
        iteration=2,
        files=[FileContent(path="main.py", content="def is_palindrome(s): return s == s[::-1]")],
    ))
    rh.executions.append(ExecutionRecord(
        iteration=2, stdout="True", stderr="", exit_code=0,
        traceback="", success=True, execution_time_ms=8.0,
    ))
    rh.token_usage.append(TokenUsage(
        iteration=2, call_type="repair",
        input_tokens=200, output_tokens=60, total_tokens=260,
    ))

    rh.finalize("success")
    return rh


# ---------------------------------------------------------------------------
# Test 1: iteration folder structure
# ---------------------------------------------------------------------------

class TestIterationFolderStructure:

    def test_run_dir_name_format(self):
        run_id = uuid4()
        name = _make_run_dir_name(run_id, "write a palindrome checker")
        assert name.startswith(f"run_{str(run_id)[:8]}_")
        assert "palindrome" in name

    def test_run_dir_name_special_chars_stripped(self):
        run_id = uuid4()
        name = _make_run_dir_name(run_id, "build a FastAPI / REST service!")
        # Special characters become underscores
        assert "/" not in name
        assert "!" not in name

    def test_save_iteration_creates_folder_and_files(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        iter_dir = p.save_iteration(sample_run, iteration=1)

        assert iter_dir.is_dir()
        assert (iter_dir / "main.py").exists()
        assert (iter_dir / "metadata.json").exists()

    def test_metadata_json_content(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        iter_dir = p.save_iteration(sample_run, iteration=1)

        meta = json.loads((iter_dir / "metadata.json").read_text())
        assert meta["iteration"] == 1
        assert meta["execution"]["success"] is False
        assert meta["error_analysis"]["category"] == "logical"
        assert meta["token_usage"]["total_tokens"] == 150

    def test_save_run_creates_both_iterations_and_summary(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        run_dir = p.save_run(sample_run)

        assert (run_dir / "run_summary.json").exists()
        assert (run_dir / "iteration_1").is_dir()
        assert (run_dir / "iteration_2").is_dir()

    def test_run_summary_json_content(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        run_dir = p.save_run(sample_run)

        summary = json.loads((run_dir / "run_summary.json").read_text())
        assert summary["final_status"] == "success"
        assert summary["total_iterations"] == 2
        assert "final_working_code_path" in summary
        assert summary["requirement"] == "write a palindrome checker"

    def test_each_iteration_has_its_own_source_file(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        p.save_run(sample_run)

        run_dir_name = _make_run_dir_name(sample_run.run_id, sample_run.requirement)
        run_dir = tmp_outputs / run_dir_name

        iter1_code = (run_dir / "iteration_1" / "main.py").read_text()
        iter2_code = (run_dir / "iteration_2" / "main.py").read_text()

        # The two iterations contain different code
        assert iter1_code != iter2_code
        assert "s[::-1]" in iter2_code  # fixed version

    def test_read_final_files(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        run_dir = p.save_run(sample_run)

        files = p.read_final_files(run_dir)
        assert len(files) == 1
        assert files[0]["path"] == "main.py"
        assert "s[::-1]" in files[0]["content"]

    def test_compute_metrics_counts_tokens_and_batch(self, tmp_outputs, sample_run):
        p = RunHistoryPersistence(tmp_outputs)
        p.save_run(sample_run)

        metrics = p.compute_metrics(sample_run.run_id)
        assert metrics["success"] is True
        assert metrics["total_input_tokens"] == 300
        assert metrics["total_output_tokens"] == 110
        assert metrics["total_llm_tokens"] == 410
        assert metrics["total_execution_calls"] == 2
        assert metrics["error_category_distribution"] == {"logical": 1}

        batch = p.compute_batch_metrics([sample_run.run_id])
        assert batch["total_runs"] == 1
        assert batch["success_rate"] == 1.0
        assert batch["first_attempt_success_rate"] == 0.0
        assert batch["avg_iterations"] == 2.0
        assert batch["median_iterations"] == 2.0
        assert batch["total_tokens"] == 410


# ---------------------------------------------------------------------------
# Test 2: embedding into run_memory
# ---------------------------------------------------------------------------

class TestRunMemoryEmbed:

    def test_embed_run_upserts_document(self, tmp_outputs, sample_run):
        """embed_run() should add a document to the run_memory collection."""
        with (
            patch("odin.rag.run_memory.chromadb") as mock_chroma,
            patch("odin.rag.run_memory.SentenceTransformer") as mock_st,
            patch("odin.rag.run_memory.RunHistoryPersistence") as mock_persistence_cls,
        ):
            # Set up mocks
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
            mock_st.return_value = mock_model

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client

            mock_persistence = MagicMock()
            mock_persistence.read_final_files.return_value = [
                {"path": "main.py", "content": "def is_palindrome(s): return s == s[::-1]"}
            ]
            mock_persistence_cls.return_value = mock_persistence

            from odin.rag.run_memory import RunMemory
            rm = RunMemory()

            run_dir = tmp_outputs / "run_abc_palindrome"
            run_dir.mkdir()
            rm.embed_run(sample_run, run_dir)

            mock_collection.upsert.assert_called_once()
            call_kwargs = mock_collection.upsert.call_args
            assert str(sample_run.run_id) in call_kwargs.kwargs["ids"]
            # Metadata should carry outcome and run_folder
            meta = call_kwargs.kwargs["metadatas"][0]
            assert meta["outcome"] == "success"
            assert "run_folder" in meta

    def test_embed_run_stores_both_success_and_failure(self, tmp_outputs):
        """Both successful and failed runs should be embedded."""
        failed_run = RunHistory(requirement="compute fibonacci")
        failed_run.finalize("failure")

        with (
            patch("odin.rag.run_memory.chromadb") as mock_chroma,
            patch("odin.rag.run_memory.SentenceTransformer") as mock_st,
            patch("odin.rag.run_memory.RunHistoryPersistence") as mock_persistence_cls,
        ):
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 384)
            mock_st.return_value = mock_model

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client

            mock_persistence = MagicMock()
            mock_persistence.read_final_files.return_value = []
            mock_persistence_cls.return_value = mock_persistence

            from odin.rag.run_memory import RunMemory
            rm = RunMemory()
            run_dir = tmp_outputs / "run_xyz_fib"
            run_dir.mkdir()
            rm.embed_run(failed_run, run_dir)

            meta = mock_collection.upsert.call_args.kwargs["metadatas"][0]
            assert meta["outcome"] == "failure"


# ---------------------------------------------------------------------------
# Test 3: two-step semantic search → read actual files from disk
# ---------------------------------------------------------------------------

class TestRunMemoryQuery:

    def _make_run_memory_with_hit(self, tmp_outputs, sample_run, similarity_distance=0.2):
        """Return a RunMemory whose ChromaDB query returns one hit."""
        run_dir = tmp_outputs / _make_run_dir_name(sample_run.run_id, sample_run.requirement)
        run_dir.mkdir(parents=True, exist_ok=True)

        with (
            patch("odin.rag.run_memory.chromadb") as mock_chroma,
            patch("odin.rag.run_memory.SentenceTransformer") as mock_st,
            patch("odin.rag.run_memory.RunHistoryPersistence") as mock_persistence_cls,
        ):
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
            mock_st.return_value = mock_model

            mock_collection = MagicMock()
            mock_collection.count.return_value = 1
            mock_collection.query.return_value = {
                "documents": [["Requirement: write a palindrome checker\nOutcome: success"]],
                "metadatas": [[{
                    "run_id": str(sample_run.run_id),
                    "outcome": "success",
                    "run_folder": str(run_dir),
                    "timestamp": datetime.now().isoformat(),
                    "error_categories": "logical",
                }]],
                "distances": [[similarity_distance]],
            }
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client

            mock_persistence = MagicMock()
            mock_persistence.read_final_files.return_value = [
                {"path": "main.py", "content": "def is_palindrome(s): return s == s[::-1]"}
            ]
            mock_persistence_cls.return_value = mock_persistence

            from odin.rag.run_memory import RunMemory
            rm = RunMemory()
            return rm, mock_persistence

    def test_query_returns_past_run_context_above_threshold(self, tmp_outputs, sample_run):
        """A similar past run above threshold returns a PastRunContext."""
        # distance=0.2 → similarity = 1 - 0.2/2 = 0.90, above threshold 0.70
        rm, mock_persistence = self._make_run_memory_with_hit(tmp_outputs, sample_run, 0.2)

        with patch("odin.rag.run_memory.config") as mock_cfg:
            mock_cfg.run_memory_similarity_threshold = 0.70
            mock_cfg.rag_embedding_model = "all-MiniLM-L6-v2"
            ctx = rm.query("check whether a string reads the same forwards and backwards")

        assert ctx is not None
        assert isinstance(ctx, PastRunContext)
        assert ctx.outcome == "success"
        assert len(ctx.final_files) == 1
        assert "s[::-1]" in ctx.final_files[0]["content"]

    def test_query_returns_none_below_threshold(self, tmp_outputs, sample_run):
        """A result below similarity threshold returns None."""
        # distance=1.6 → similarity = 1 - 1.6/2 = 0.20, below threshold 0.70
        rm, _ = self._make_run_memory_with_hit(tmp_outputs, sample_run, 1.6)

        with patch("odin.rag.run_memory.config") as mock_cfg:
            mock_cfg.run_memory_similarity_threshold = 0.70
            mock_cfg.rag_embedding_model = "all-MiniLM-L6-v2"
            ctx = rm.query("compute fibonacci sequence")

        assert ctx is None

    def test_two_step_read_calls_read_final_files(self, tmp_outputs, sample_run):
        """Verify the two-step: query → resolve folder → read files from disk."""
        rm, mock_persistence = self._make_run_memory_with_hit(tmp_outputs, sample_run, 0.2)

        with patch("odin.rag.run_memory.config") as mock_cfg:
            mock_cfg.run_memory_similarity_threshold = 0.70
            mock_cfg.rag_embedding_model = "all-MiniLM-L6-v2"
            ctx = rm.query("palindrome check")

        # read_final_files must have been called with the run_folder from metadata
        mock_persistence.read_final_files.assert_called_once()

    def test_query_returns_none_when_collection_empty(self):
        """Empty run_memory returns None without querying."""
        with (
            patch("odin.rag.run_memory.chromadb") as mock_chroma,
            patch("odin.rag.run_memory.SentenceTransformer") as mock_st,
            patch("odin.rag.run_memory.RunHistoryPersistence"),
        ):
            mock_model = MagicMock()
            mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 384)
            mock_st.return_value = mock_model

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0  # empty
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client

            from odin.rag.run_memory import RunMemory
            rm = RunMemory()
            ctx = rm.query("anything")

        assert ctx is None
        mock_collection.query.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: re-prompt (is_continuation) does NOT trigger run_memory
# ---------------------------------------------------------------------------

class TestContinuationDoesNotQueryRunMemory:

    @pytest.mark.asyncio
    async def test_continue_run_skips_run_memory_query(self, sample_run):
        """backend.continue_run() must not call run_memory.query()."""
        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
            patch("odin.backend.RunMemory") as mock_rm_cls,
        ):
            # Graph that succeeds cleanly
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"success": True})
            mock_build.return_value = mock_graph

            mock_persistence = MagicMock()
            mock_persistence.save_run.return_value = Path("/tmp/fake_run")
            mock_pers_cls.return_value = mock_persistence

            mock_run_memory = MagicMock()
            mock_rm_cls.return_value = mock_run_memory

            # Patch config so backend doesn't need a real API key
            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 5
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                from odin.backend import OdinBackend
                backend = OdinBackend()

                # Ensure the run has a final_status so finalize isn't called again
                sample_run.finalize("success")
                await backend.continue_run(sample_run, user_feedback="make it handle None")

        # The critical assertion: run_memory.query() must NOT have been called
        mock_run_memory.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_run_does_query_run_memory(self):
        """backend.run_pipeline() MUST call run_memory.query() for new requests."""
        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
            patch("odin.backend.RunMemory") as mock_rm_cls,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"success": True})
            mock_build.return_value = mock_graph

            mock_persistence = MagicMock()
            mock_persistence.save_run.return_value = Path("/tmp/fake_run")
            mock_pers_cls.return_value = mock_persistence

            mock_run_memory = MagicMock()
            mock_run_memory.query.return_value = None  # no match
            mock_rm_cls.return_value = mock_run_memory

            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 5
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                from odin.backend import OdinBackend
                backend = OdinBackend()
                await backend.run_pipeline("reverse a linked list")

        # For a new request, run_memory.query() must have been called exactly once
        mock_run_memory.query.assert_called_once_with("reverse a linked list")

    def test_graph_state_is_continuation_false_for_new_run(self):
        """For run_pipeline(), the initial state must have is_continuation=False."""
        captured_states = []

        async def fake_ainvoke(state):
            captured_states.append(dict(state))

        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
            patch("odin.backend.RunMemory") as mock_rm_cls,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_build.return_value = mock_graph

            mock_persistence = MagicMock()
            mock_persistence.save_run.return_value = Path("/tmp/x")
            mock_pers_cls.return_value = mock_persistence

            mock_rm = MagicMock()
            mock_rm.query.return_value = None
            mock_rm_cls.return_value = mock_rm

            import asyncio
            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 5
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                from odin.backend import OdinBackend
                backend = OdinBackend()
                asyncio.run(backend.run_pipeline("check palindrome"))

        assert captured_states, "ainvoke was never called"
        state = captured_states[0]
        assert state["is_continuation"] is False
        assert state["past_run_context"] is None  # no match was returned

    def test_graph_state_is_continuation_true_for_continue_run(self, sample_run):
        """For continue_run(), the initial state must have is_continuation=True."""
        captured_states = []

        async def fake_ainvoke(state):
            captured_states.append(dict(state))

        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
            patch("odin.backend.RunMemory") as mock_rm_cls,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_build.return_value = mock_graph

            mock_persistence = MagicMock()
            mock_persistence.save_run.return_value = Path("/tmp/x")
            mock_pers_cls.return_value = mock_persistence

            mock_rm_cls.return_value = MagicMock()

            import asyncio
            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 5
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                sample_run.finalize("success")
                from odin.backend import OdinBackend
                backend = OdinBackend()
                asyncio.run(backend.continue_run(sample_run, "please handle None input"))

        assert captured_states
        state = captured_states[0]
        assert state["is_continuation"] is True
        assert state["user_feedback"] == "please handle None input"
        assert state["past_run_context"] is None

    @pytest.mark.asyncio
    async def test_backend_with_run_memory_disabled(self):
        """Setting use_run_memory=False avoids querying or embedding run_memory."""
        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
            patch("odin.backend.RunMemory") as mock_rm_cls,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value={"success": True})
            mock_build.return_value = mock_graph

            mock_persistence = MagicMock()
            mock_persistence.save_run.return_value = Path("/tmp/fake_run")
            mock_pers_cls.return_value = mock_persistence

            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 5
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                from odin.backend import OdinBackend
                backend = OdinBackend(use_run_memory=False)
                rh = await backend.run_pipeline("reverse a linked list")

        mock_rm_cls.assert_not_called()
        assert rh.final_status == "success"

    @pytest.mark.asyncio
    async def test_backend_records_error_message_on_exception(self, tmp_outputs):
        """API/pipeline exceptions are captured in run_history.error_message and persisted."""
        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("503 UNAVAILABLE"))
            mock_build.return_value = mock_graph

            real_persistence = RunHistoryPersistence(tmp_outputs)
            mock_pers_cls.return_value = real_persistence

            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 3
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                from odin.backend import OdinBackend
                backend = OdinBackend(use_run_memory=False)
                rh = await backend.run_pipeline("test failure diagnostics")

        assert rh.final_status == "failure"
        assert rh.error_message == "RuntimeError: 503 UNAVAILABLE"
        reloaded = real_persistence.load_run(rh.run_id)
        assert reloaded.error_message == "RuntimeError: 503 UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_continue_run_preserves_and_updates_test_input(self, sample_run):
        """continue_run() preserves existing test_input instead of wiping it to empty string."""
        captured_states = []

        async def fake_ainvoke(state):
            captured_states.append(dict(state))
            return {"success": True}

        with (
            patch("odin.backend.build_graph") as mock_build,
            patch("odin.backend.RunHistoryPersistence") as mock_pers_cls,
        ):
            mock_graph = MagicMock()
            mock_graph.ainvoke = fake_ainvoke
            mock_build.return_value = mock_graph

            mock_persistence = MagicMock()
            mock_persistence.save_run.return_value = Path("/tmp/x")
            mock_pers_cls.return_value = mock_persistence

            with patch("odin.backend.config") as mock_cfg:
                mock_cfg.max_iterations = 3
                mock_cfg.execution_timeout = 5
                mock_cfg.validate_config = MagicMock()
                mock_cfg.ensure_directories = MagicMock()

                sample_run.test_input = "racecar\n"
                from odin.backend import OdinBackend
                backend = OdinBackend(use_run_memory=False)
                await backend.continue_run(sample_run, "handle uppercase")

        assert captured_states[0]["test_input"] == "racecar\n"
        # Fresh continuation budget: current_iteration (2) + max_iterations (3) = 5
        assert captured_states[0]["max_iterations"] == 5

