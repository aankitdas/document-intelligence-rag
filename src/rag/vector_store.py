"""
Vector Store Module
===================

Purpose: Store embeddings and retrieve similar ones

This module uses Chroma for persistent, efficient vector storage.
Chroma is free, local, and production-ready.

Key Concepts:
  • Vector storage: Persistent storage mapping chunk_id → embedding
  • Metadata: Source info, text preview, etc.
  • Retrieval: Find top-k most similar vectors using cosine similarity
  • Persistence: Data survives application restarts
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging
import chromadb
import os

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChromaVectorStore:
    """
    Vector store using Chroma (persistent, free, production-ready).
    
    Chroma is a modern vector database that:
    • Stores embeddings persistently on disk
    • Provides similarity search
    • Is completely free and open source
    • Works locally (no API calls)
    
    This is the recommended implementation for production RAG systems.
    """
    
    def __init__(self, persist_directory: str = ".chromadb", collection_name: str = "rag"):
        """
        Initialize Chroma vector store.
        
        Args:
            persist_directory: Where to store vectors on disk
            collection_name: Name of the collection (namespace)
        
        Example:
            >>> store = ChromaVectorStore(persist_directory="./data/vectors")
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(
                f"✓ Initialized Chroma vector store at {persist_directory} "
                f"(collection: {collection_name})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.client.persist()
            self.client.shutdown()
        except Exception:
            pass

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a chunk with its embedding to the store.
        
        Args:
            chunk_id: Unique identifier for chunk
            text: Original text content
            embedding: Vector representation (list of floats)
            metadata: Optional metadata (source, page number, etc.)
        
        Example:
            >>> store.add(
            ...     "doc1_chunk_0",
            ...     "Machine learning is AI",
            ...     [0.1, 0.2, ..., 0.384],
            ...     metadata={"doc_id": "doc1", "page": 1}
            ... )
        """
        try:
            self.collection.add(
                ids=[chunk_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata or {}]
            )
            logger.debug(f"Added chunk {chunk_id} ({len(text)} chars)")
        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id}: {e}")
            raise
    
    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Find most similar chunks to query.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            List of RetrievalResult objects, sorted by similarity (highest first)
        
        Example:
            >>> results = store.retrieve(query_embedding, top_k=3)
            >>> for r in results:
            ...     print(f"{r.similarity:.3f} | {r.text[:60]}")
        """
        try:
            if self.collection.count() == 0:
                logger.warning("Vector store is empty")
                return []
            
            # Query Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results["ids"] or not results["ids"][0]:
                logger.debug("No results found for query")
                return []
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            
            for i, chunk_id in enumerate(results["ids"][0]):
                # Chroma returns distances, convert to similarity (1 - distance for cosine)
                # Note: Chroma with cosine metric returns distances
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    similarity=similarity,
                    metadata=results["metadatas"][0][i]
                )
                retrieval_results.append(result)
            
            logger.debug(f"Retrieved {len(retrieval_results)} chunks")
            return retrieval_results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def size(self) -> int:
        """Return number of chunks in store."""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"Failed to get store size: {e}")
            return 0
    
    def delete(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the store.
        
        Args:
            chunk_id: ID of chunk to delete
        
        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[chunk_id])
            logger.debug(f"Deleted chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all vectors from store."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
            raise





# ============ TESTS ============

import tempfile
import shutil
import time

def test_chroma_vector_store():
    temp_dir = tempfile.mkdtemp()
    
    store = ChromaVectorStore(persist_directory=temp_dir)
    
    try:
        # Add chunks
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.9, 0.1, 0.0]
        vec3 = [0.0, 1.0, 0.0]
        
        store.add("chunk1", "Machine learning", vec1, metadata={"source": "test"})
        store.add("chunk2", "Deep learning networks", vec2, metadata={"source": "test"})
        store.add("chunk3", "Cooking recipes", vec3, metadata={"source": "test"})
        
        # Retrieve
        results = store.retrieve(vec1, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"
        print("✓ Chroma test passed!")
    
    finally:
        # Cleanup Chroma resources
        try:
            if hasattr(store, "client"):
                store.client.close()
                del store.client
                del store.collection
        except Exception as e:
            logger.warning(f"Error closing Chroma client: {e}")
        
        # Give Windows time to release file handles
        time.sleep(1.0)
        
        # Retry logic for Windows file deletion
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                shutil.rmtree(temp_dir)
                break
            except PermissionError:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(0.5)
                else:
                    logger.warning(f"Could not delete temp directory {temp_dir}, skipping")
                    break



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test Chroma
    try:
        test_chroma_vector_store()
    except ImportError:
        print("Chroma not installed, skipping test")
    
    # Test SimpleVectorStore
    test_simple_vector_store()