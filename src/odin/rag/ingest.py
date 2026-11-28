"""Standalone script for ingesting knowledge into ChromaDB.

This script chunks and embeds documentation, API references, and curated
error-fix examples into ChromaDB for runtime retrieval by ODIN.

Run this independently before using the main ODIN application.

Usage:
    python -m odin.rag.ingest [--source-dir PATH]

Chunking strategy:
- Documentation: Split by sections/headers, with 500-token chunks and 50-token overlap
- API references: One chunk per function/class/method
- Error examples: One chunk per example (error + fix pair)

Metadata attached to each chunk:
- source: File path or URL
- category: Which error categories this knowledge helps with
- doc_type: "documentation", "api_reference", or "example"
- library: Relevant library name (if applicable)
"""
import argparse
import logging
import re
from pathlib import Path
from typing import Any, List

from odin.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by tokens (approximated by words).
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens (approximated as words)
        overlap: Overlap between chunks in tokens
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def ingest_documentation(
    source_dir: Path,
    collection,
    embedding_model: Any,
    exclude_files: set[Path] | None = None,
) -> int:
    """Ingest documentation files from a directory.
    
    Expects markdown or text files with documentation content.
    
    Args:
        source_dir: Directory containing documentation files
        collection: ChromaDB collection
        embedding_model: Sentence transformer model
        exclude_files: Optional set of resolved file paths to skip (e.g. examples.md)
    
    Returns:
        Number of chunks ingested
    """
    logger.info(f"Ingesting documentation from {source_dir}")
    
    if not source_dir.exists():
        logger.warning(f"Source directory {source_dir} does not exist")
        return 0
    
    excluded = {p.resolve() for p in (exclude_files or set())}
    chunks_ingested = 0
    
    for doc_file in source_dir.rglob("*.md"):
        if doc_file.resolve() in excluded:
            logger.info(f"Skipping {doc_file} (ingested separately)")
            continue
        logger.info(f"Processing {doc_file}")
        
        content = doc_file.read_text(encoding="utf-8")
        chunks = chunk_text(content, chunk_size=500, overlap=50)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_file.stem}_{i}"
            embedding = embedding_model.encode(chunk, convert_to_numpy=True).tolist()
            
            metadata = {
                "source": str(doc_file),
                "doc_type": "documentation",
                "chunk_index": i,
                "library": _infer_library(doc_file.name),
            }
            
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata],
            )
            
            chunks_ingested += 1
    
    logger.info(f"Ingested {chunks_ingested} documentation chunks")
    return chunks_ingested


def ingest_examples(
    examples_file: Path,
    collection,
    embedding_model: Any,
) -> int:
    """Ingest curated error-fix examples.
    
    Expects a file with examples in the format:
    
    # Example: <description>
    Category: <category>
    Error:
    ```
    <error message>
    ```
    Fix:
    ```
    <fixed code>
    ```
    
    Args:
        examples_file: File containing examples
        collection: ChromaDB collection
        embedding_model: Sentence transformer model
    
    Returns:
        Number of examples ingested
    """
    logger.info(f"Ingesting examples from {examples_file}")
    
    if not examples_file.exists():
        logger.warning(f"Examples file {examples_file} does not exist")
        return 0
    
    content = examples_file.read_text(encoding="utf-8")
    
    # Parse examples
    example_pattern = r"# Example: (.+?)\nCategory: (.+?)\nError:\s*```(.+?)```\s*Fix:\s*```(.+?)```"
    examples = re.findall(example_pattern, content, re.DOTALL)
    
    for i, (description, category, error, fix) in enumerate(examples):
        example_text = f"Error: {error.strip()}\n\nFix: {fix.strip()}"
        example_id = f"example_{i}"
        
        embedding = embedding_model.encode(example_text, convert_to_numpy=True).tolist()
        
        metadata = {
            "source": str(examples_file),
            "doc_type": "example",
            "category": category.strip(),
            "description": description.strip(),
        }
        
        collection.upsert(
            ids=[example_id],
            embeddings=[embedding],
            documents=[example_text],
            metadatas=[metadata],
        )
    
    logger.info(f"Ingested {len(examples)} examples")
    return len(examples)


def ingest_python_stdlib_docs(
    collection,
    embedding_model: Any,
) -> int:
    """Ingest basic Python standard library documentation snippets.
    
    This creates a small set of essential stdlib docs as a starting point.
    In a production system, you'd scrape/download official Python docs.
    
    Returns:
        Number of chunks ingested
    """
    logger.info("Ingesting Python stdlib documentation snippets")
    
    # Basic stdlib documentation snippets
    stdlib_docs = [
        {
            "content": "The itertools module provides functions for creating iterators for efficient looping. Key functions: chain(), combinations(), permutations(), product(), cycle(), repeat(), groupby().",
            "library": "itertools",
            "category": "uncommon_pattern",
        },
        {
            "content": "functools.reduce(function, iterable, initializer=None) applies a function cumulatively to items in an iterable. Example: reduce(lambda x, y: x+y, [1,2,3,4,5]) returns 15.",
            "library": "functools",
            "category": "uncommon_pattern",
        },
        {
            "content": "asyncio provides async/await syntax for concurrent programming. Use async def to define coroutines, await to call them, asyncio.run() to execute, asyncio.gather() for concurrent execution.",
            "library": "asyncio",
            "category": "uncommon_pattern",
        },
        {
            "content": "collections.abc provides abstract base classes for containers. Use Sequence, MutableSequence, Mapping, MutableMapping, Set, MutableSet to type-hint generic collections.",
            "library": "collections.abc",
            "category": "uncommon_pattern",
        },
        {
            "content": "pandas DataFrame.append() was deprecated in pandas 2.0 and removed. Use pd.concat([df1, df2]) instead. The append method on DataFrame no longer exists.",
            "library": "pandas",
            "category": "version_specific",
        },
        {
            "content": "numpy.matrix class is deprecated since NumPy 1.20. Use numpy.ndarray or numpy.array instead. Operations on matrix objects should be converted to use array.",
            "library": "numpy",
            "category": "version_specific",
        },
        {
            "content": "requests.get(url, params=dict, headers=dict, timeout=seconds) makes HTTP GET requests. Returns Response object with .status_code, .text, .json() methods. Always set timeout to avoid hanging.",
            "library": "requests",
            "category": "library_api",
        },
        {
            "content": "pandas.read_csv(filepath, sep=',', header='infer', names=list, dtype=dict, parse_dates=list) reads CSV files. Common errors: FileNotFoundError (check path), ParserError (check sep), KeyError (check column names).",
            "library": "pandas",
            "category": "library_api",
        },
    ]
    
    for i, doc in enumerate(stdlib_docs):
        doc_id = f"stdlib_{i}"
        embedding = embedding_model.encode(doc["content"], convert_to_numpy=True).tolist()
        
        metadata = {
            "source": "builtin_stdlib_docs",
            "doc_type": "documentation",
            "library": doc["library"],
            "category": doc["category"],
        }
        
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc["content"]],
            metadatas=[metadata],
        )
    
    logger.info(f"Ingested {len(stdlib_docs)} stdlib documentation snippets")
    return len(stdlib_docs)


def _infer_library(filename: str) -> str:
    """Infer library name from filename."""
    filename_lower = filename.lower()
    
    libraries = ["pandas", "numpy", "requests", "flask", "django", "fastapi", "torch"]
    for lib in libraries:
        if lib in filename_lower:
            return lib
    
    return "python"


def main():
    """Main ingestion entry point."""
    parser = argparse.ArgumentParser(description="Ingest knowledge into ODIN's RAG system")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("./knowledge"),
        help="Directory containing documentation to ingest",
    )
    parser.add_argument(
        "--examples-file",
        type=Path,
        default=Path("./knowledge/examples.md"),
        help="File containing curated error-fix examples",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before ingesting",
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    config.ensure_directories()

    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    
    # Initialize embedding model
    logger.info(f"Loading embedding model: {config.rag_embedding_model}")
    embedding_model = SentenceTransformer(config.rag_embedding_model)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=str(config.rag_db_path),
        settings=Settings(anonymized_telemetry=False),
    )
    
    # Create or get collection
    collection_name = config.knowledge_collection_name
    if args.reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection")
        except Exception:
            pass
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Python documentation and code examples for ODIN"}
    )
    
    # Ingest from various sources
    total_chunks = 0
    
    # Always ingest built-in stdlib docs
    total_chunks += ingest_python_stdlib_docs(collection, embedding_model)
    
    # Ingest documentation if source dir exists (excluding examples_file to avoid duplicates)
    if args.source_dir.exists():
        exclude = {args.examples_file} if args.examples_file.exists() else set()
        total_chunks += ingest_documentation(
            args.source_dir, collection, embedding_model, exclude_files=exclude
        )
    else:
        logger.info(f"Source directory {args.source_dir} not found, skipping documentation ingest")
    
    # Ingest examples if file exists
    if args.examples_file.exists():
        total_chunks += ingest_examples(args.examples_file, collection, embedding_model)
    else:
        logger.info(f"Examples file {args.examples_file} not found, skipping examples ingest")
    
    logger.info(f"Ingestion complete! Total chunks: {total_chunks}")
    logger.info(f"Collection now has {collection.count()} documents")


if __name__ == "__main__":
    main()
