"""
RAG Package
===========

Modular Retrieval-Augmented Generation system with Ollama + Groq
"""

from .chunker import chunk_text, Chunk, chunk_documents
from .embeddings import OllamaEmbeddingClient, cosine_similarity
from .vector_store import ChromaVectorStore, RetrievalResult
from .llm import GroqLLMClient, build_context_string
from .pdf_processor import PDFProcessor
from .pipeline import RAGPipeline, RAGConfig

__all__ = [
    # Chunking
    "chunk_text",
    "Chunk",
    "chunk_documents",
    # Embeddings
    "OllamaEmbeddingClient",
    "cosine_similarity",
    # Vector Store
    "ChromaVectorStore",
    "SimpleVectorStore",
    "RetrievalResult",
    # LLM
    "GroqLLMClient",
    "build_context_string",
    # PDF Processing
    "PDFProcessor",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
]

__version__ = "0.1.0"