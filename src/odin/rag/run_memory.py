"""Past-run memory via a dedicated ChromaDB collection.

This is entirely separate from the `odin_knowledge` collection that holds
static documentation and API-reference chunks.  That collection is for
debugging help during a run; this one is for learning from past runs.

Collection name:  run_memory  (configured in config.run_memory_collection_name)
Storage path:     ./chroma_data/  (shared PersistentClient with odin_knowledge)

What gets embedded
------------------
One document per completed run (success OR failure — both are stored so the
system can learn "this approach didn't work either").  The embedded text is a
composed summary:

    Requirement: <original NL requirement>
    Outcome: <success | failure | max_iterations_reached>
    Error categories encountered: <comma-separated list>
    Final code:
    <content of every file in the last iteration, concatenated>

Metadata stored alongside the embedding
----------------------------------------
    run_id          : str (UUID)
    outcome         : str
    run_folder      : str  (absolute path to outputs/run_<id>_<slug>/)
    timestamp       : ISO-8601 str
    error_categories: comma-separated str (for filtering if desired)

Two-step retrieval
------------------
1. Semantic search: embed the new requirement → query run_memory →
   returns top-k matches with similarity scores.
2. Threshold gate: discard any match below config.run_memory_similarity_threshold.
3. For matches that pass: call persistence.read_final_files(run_folder) to read
   the real source files from disk.
4. Return a PastRunContext object (or None if no match passes threshold).

Why two steps instead of storing code in the vector store?
   Embedding large blobs of code into ChromaDB would balloon the index and make
   similarity scores less reliable.  The vector store just tells us *which*
   past run is relevant; the folder holds the actual content.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from odin.config import config
from odin.persistence import RunHistoryPersistence
from odin.state import PastRunContext, RunHistory

logger = logging.getLogger(__name__)

# Lazily loaded on first RunMemory instantiation so non-memory code paths
# (and unit tests that patch these module attributes) avoid PyTorch import overhead.
chromadb: Any = None
SentenceTransformer: Any = None


class RunMemory:
    """Reads and writes the run_memory ChromaDB collection.

    A single instance is created by OdinBackend and reused across calls so
    the embedding model is loaded only once.
    """

    def __init__(self) -> None:
        st_cls = SentenceTransformer
        if st_cls is None:
            from sentence_transformers import SentenceTransformer as st_cls
        self._model = st_cls(config.rag_embedding_model)

        chroma_mod = chromadb
        if chroma_mod is None:
            import chromadb as chroma_mod
            from chromadb.config import Settings
            client_settings = Settings(anonymized_telemetry=False)
        else:
            client_settings = chroma_mod.config.Settings(anonymized_telemetry=False)

        self._client = chroma_mod.PersistentClient(
            path=str(config.chroma_data_path),
            settings=client_settings,
        )
        self._collection = self._client.get_or_create_collection(
            name=config.run_memory_collection_name,
            metadata={"description": "Past ODIN run summaries for memory-augmented generation"},
        )
        self._persistence = RunHistoryPersistence()
        logger.info(
            "RunMemory ready — %d past runs in collection",
            self._collection.count(),
        )

    # ── Writing ─────────────────────────────────────────────────────────────

    def embed_run(self, run_history: RunHistory, run_folder: str | Path) -> None:
        """Embed a completed run into the run_memory collection.

        Called once per run at the end of backend.run_pipeline(), after
        run_summary.json has been written.  Both successful and failed runs
        are stored (tagged via the `outcome` metadata field).

        Args:
            run_history:  The completed RunHistory object.
            run_folder:   Path to the run's output folder on disk.
        """
        run_folder = Path(run_folder)

        # Read final code files from disk for the summary text
        final_files = self._persistence.read_final_files(run_folder)
        code_block = "\n\n".join(
            f"# --- {f['path']} ---\n{f['content']}" for f in final_files
        )

        error_categories = list(
            dict.fromkeys(ea.category for ea in run_history.error_analyses)
        )

        summary_text = (
            f"Requirement: {run_history.requirement}\n"
            f"Outcome: {run_history.final_status}\n"
            f"Error categories encountered: {', '.join(error_categories) or 'none'}\n"
            f"Final code:\n{code_block}"
        )

        embedding = self._model.encode(summary_text, convert_to_numpy=True).tolist()
        doc_id = str(run_history.run_id)

        metadata: dict[str, Any] = {
            "run_id": doc_id,
            "outcome": run_history.final_status or "unknown",
            "run_folder": str(run_folder.resolve()),
            "timestamp": run_history.end_time.isoformat() if run_history.end_time else "",
            "error_categories": ", ".join(error_categories),
        }

        # Upsert so re-running the same UUID (e.g. during testing) doesn't error
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[summary_text],
            metadatas=[metadata],
        )
        logger.info("Embedded run %s (outcome=%s) into run_memory", doc_id, run_history.final_status)

    # ── Reading ─────────────────────────────────────────────────────────────

    def query(self, new_requirement: str) -> PastRunContext | None:
        """Search for a similar past run.

        Implements the two-step lookup:
        1. Embed the new requirement and query run_memory.
        2. Check the top result against config.run_memory_similarity_threshold.
        3. If it passes, read the actual source files from disk.
        4. Return a populated PastRunContext, or None if no match.

        Args:
            new_requirement: The user's new NL task description.

        Returns:
            A PastRunContext if a sufficiently similar past run is found,
            otherwise None.
        """
        if self._collection.count() == 0:
            logger.debug("run_memory is empty — skipping lookup")
            return None

        t0 = time.perf_counter()
        embedding = self._model.encode(new_requirement, convert_to_numpy=True).tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        if not results["documents"] or not results["documents"][0]:
            logger.debug("run_memory query returned no results")
            return None

        distance: float = results["distances"][0][0]
        # ChromaDB uses squared L2 distance by default; convert to cosine-like
        # similarity in [0, 1].  For unit-normalised vectors, L2² = 2(1−cos),
        # so cos_sim = 1 − dist/2.  all-MiniLM-L6-v2 outputs unit vectors.
        similarity: float = max(0.0, 1.0 - distance / 2.0)

        threshold = config.run_memory_similarity_threshold
        logger.info(
            "run_memory top-1 similarity=%.3f (threshold=%.2f, latency=%.1fms)",
            similarity,
            threshold,
            latency_ms,
        )

        if similarity < threshold:
            logger.info("No past run meets similarity threshold — proceeding without memory context")
            return None

        meta: dict[str, Any] = results["metadatas"][0][0]
        summary_text: str = results["documents"][0][0]
        run_folder = meta.get("run_folder", "")

        # Step 2: read actual files from disk
        final_files: list[dict[str, str]] = []
        if run_folder and Path(run_folder).exists():
            final_files = self._persistence.read_final_files(run_folder)
        else:
            logger.warning("run_folder %r not found on disk — returning summary only", run_folder)

        error_cats = [c.strip() for c in meta.get("error_categories", "").split(",") if c.strip()]

        ctx = PastRunContext(
            run_id=meta.get("run_id", ""),
            requirement=_extract_field(summary_text, "Requirement"),
            outcome=meta.get("outcome", "unknown"),  # type: ignore[arg-type]
            similarity_score=similarity,
            run_folder=run_folder,
            error_categories_seen=error_cats,
            final_files=final_files,
            summary_text=summary_text,
        )
        logger.info(
            "Found similar past run %s (outcome=%s, similarity=%.3f) with %d file(s)",
            ctx.run_id,
            ctx.outcome,
            ctx.similarity_score,
            len(ctx.final_files),
        )
        return ctx

    def count(self) -> int:
        """Return number of runs stored in run_memory."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_field(text: str, field: str) -> str:
    """Extract 'Field: <value>' from the embedded summary text."""
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return line[len(field) + 1:].strip()
    return ""
