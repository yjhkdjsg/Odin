"""Unit tests for RAG ingestion and retrieval fixes."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from odin.rag.ingest import ingest_documentation, ingest_examples, ingest_python_stdlib_docs
from odin.rag.retriever import RAGRetriever


def test_ingest_documentation_excludes_examples_and_uses_upsert(tmp_path: Path):
    docs_dir = tmp_path / "knowledge"
    docs_dir.mkdir()
    doc_file = docs_dir / "pandas_guide.md"
    doc_file.write_text("Use pd.concat instead of DataFrame.append in pandas 2.0.", encoding="utf-8")
    examples_file = docs_dir / "examples.md"
    examples_file.write_text(
        "# Example: Test\nCategory: library_api\nError:\n```\nKeyError\n```\nFix:\n```\nUse dict.get\n```\n",
        encoding="utf-8",
    )

    mock_collection = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

    count = ingest_documentation(
        docs_dir, mock_collection, mock_model, exclude_files={examples_file}
    )

    assert count == 1
    mock_collection.upsert.assert_called_once()
    mock_collection.add.assert_not_called()
    assert mock_collection.upsert.call_args.kwargs["ids"] == ["pandas_guide_0"]


def test_ingest_examples_and_stdlib_use_upsert(tmp_path: Path):
    examples_file = tmp_path / "examples.md"
    examples_file.write_text(
        "# Example: Pandas append\nCategory: version_specific\nError:\n```\nAttributeError\n```\nFix:\n```\npd.concat\n```\n",
        encoding="utf-8",
    )

    mock_collection = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

    ex_count = ingest_examples(examples_file, mock_collection, mock_model)
    stdlib_count = ingest_python_stdlib_docs(mock_collection, mock_model)

    assert ex_count == 1
    assert stdlib_count == 8
    assert mock_collection.upsert.call_count == 9
    mock_collection.add.assert_not_called()


def test_retriever_handles_empty_collection():
    with (
        patch("odin.rag.retriever.chromadb") as mock_chroma,
        patch("odin.rag.retriever.SentenceTransformer") as mock_st,
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
        mock_st.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.name = "odin_knowledge"
        mock_collection.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        retriever = RAGRetriever()
        record = retriever.retrieve("AttributeError", category="library_api", iteration=1)

    assert record.triggered is True
    assert record.chunks == []
    mock_collection.query.assert_not_called()


def test_retriever_caps_n_results_and_computes_cosine_similarity():
    with (
        patch("odin.rag.retriever.chromadb") as mock_chroma,
        patch("odin.rag.retriever.SentenceTransformer") as mock_st,
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
        mock_st.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.name = "odin_knowledge"
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["Use pd.concat"]],
            "metadatas": [[{"library": "pandas"}]],
            "distances": [[0.4]],  # squared L2 -> cos_sim = 1 - 0.4/2 = 0.8
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.PersistentClient.return_value = mock_client

        retriever = RAGRetriever()
        record = retriever.retrieve("AttributeError", category="version_specific", iteration=1, top_k=5)

    assert mock_collection.query.call_args.kwargs["n_results"] == 1
    assert len(record.chunks) == 1
    assert abs(record.chunks[0]["similarity_score"] - 0.8) < 1e-6
