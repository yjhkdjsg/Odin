"""RAG retrieval using ChromaDB and Sentence Transformers.

This module provides the runtime retrieval interface for the ODIN framework.
It queries a pre-built ChromaDB collection to find relevant documentation,
examples, and API references given an error context.
"""
import logging
import time
from typing import Any

from odin.config import config
from odin.state import ErrorCategory, RetrievalRecord

logger = logging.getLogger(__name__)

# Lazily loaded on first RAGRetriever instantiation so non-RAG code paths
# (and unit tests that mock these attributes) avoid heavy PyTorch startup costs.
chromadb: Any = None
SentenceTransformer: Any = None


class RAGRetriever:
    """Retrieves relevant knowledge from ChromaDB."""
    
    def __init__(self):
        """Initialize the retriever with embedding model and ChromaDB client."""
        st_cls = SentenceTransformer
        if st_cls is None:
            from sentence_transformers import SentenceTransformer as st_cls

        self.embedding_model = st_cls(config.rag_embedding_model)
        
        # Initialize ChromaDB client
        chroma_mod = chromadb
        if chroma_mod is None:
            import chromadb as chroma_mod
            from chromadb.config import Settings
            client_settings = Settings(anonymized_telemetry=False)
        else:
            client_settings = chroma_mod.config.Settings(anonymized_telemetry=False)

        self.client = chroma_mod.PersistentClient(
            path=str(config.rag_db_path),
            settings=client_settings,
        )
        
        # Get or create the collection
        collection_name = config.knowledge_collection_name
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Connected to existing collection with {self.collection.count()} documents")
        except Exception as e:
            logger.warning(f"Collection not found or error: {e}. Creating empty collection.")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Python documentation and code examples for ODIN"}
            )
    
    def retrieve(
        self,
        query: str,
        category: ErrorCategory,
        iteration: int,
        top_k: int | None = None,
    ) -> RetrievalRecord:
        """Retrieve relevant knowledge for an error.
        
        Args:
            query: The error message or traceback to search for
            category: The error category (used for metadata filtering if available)
            iteration: Current iteration number
            top_k: Number of results to return (defaults to config.rag_top_k)
        
        Returns:
            RetrievalRecord with retrieved chunks and metadata
        """
        start_time = time.perf_counter()
        top_k = top_k or config.rag_top_k

        doc_count = self.collection.count()
        if doc_count == 0:
            logger.warning("Knowledge collection '%s' is empty — returning no chunks", self.collection.name)
            return RetrievalRecord(
                iteration=iteration,
                triggered=True,
                decision_reason=f"Retrieval triggered for category '{category}' (empty knowledge base)",
                query=query,
                chunks=[],
                retrieval_latency_ms=(time.perf_counter() - start_time) * 1000,
                total_tokens=0,
            )
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        # Query ChromaDB (cap n_results to available document count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, doc_count),
            include=["documents", "metadatas", "distances"],
        )
        
        end_time = time.perf_counter()
        retrieval_latency_ms = (end_time - start_time) * 1000
        
        # Format results
        chunks = []
        total_tokens = 0
        
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunk_info = {
                    "content": doc,
                    "metadata": metadata or {},
                    # ChromaDB default L2² on unit-normalised MiniLM vectors: cos_sim = 1 - dist/2
                    "similarity_score": max(0.0, 1.0 - distance / 2.0),
                }
                chunks.append(chunk_info)
                
                # Approximate token count (rough heuristic: 1 token ≈ 4 chars)
                total_tokens += len(doc) // 4
        
        logger.info(
            f"Retrieved {len(chunks)} chunks for category '{category}' "
            f"in {retrieval_latency_ms:.2f}ms"
        )
        
        return RetrievalRecord(
            iteration=iteration,
            triggered=True,
            decision_reason=f"Retrieval triggered for category '{category}'",
            query=query,
            chunks=chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            total_tokens=total_tokens,
        )
    
    def health_check(self) -> dict[str, Any]:
        """Check if the retriever is operational.
        
        Returns:
            Dictionary with health status and statistics
        """
        try:
            doc_count = self.collection.count()
            return {
                "status": "healthy",
                "document_count": doc_count,
                "collection_name": self.collection.name,
                "embedding_model": config.rag_embedding_model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
