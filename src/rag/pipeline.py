"""
RAG Pipeline
------------
Purpose: DO all RAG stuff in to a unified pipeline
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

from .chunker import chunk_text
from .vector_store import ChromaVectorStore
from .llm import GroqLLMClient, build_context_string
from .pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env():
    """Load environment variables from project root .env file."""
    env_paths = [
        os.path.join(os.path.dirname(__file__), '../..', '.env'),
        os.path.join(os.path.dirname(__file__), '.env'),
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.debug(f"Loaded .env from: {env_path}")
            return env_path
    
    logger.warning("No .env file found")
    return None


def get_embeddings_client():
    """
    Get embeddings client based on EMBEDDING_BACKEND env var.
    
    Environment Variables:
        EMBEDDING_BACKEND: "ollama" or "sentence-transformers" (default)
        OLLAMA_BASE_URL: URL for Ollama (default: http://localhost:11434)
    
    Returns:
        Embeddings client instance
    """
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
    
    if backend == "ollama":
        logger.info("Using Ollama embeddings")
        from .embeddings import OllamaEmbeddingClient
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddingClient(
            base_url=base_url,
            model="nomic-embed-text"
        )
    else:
        # sentence-transformers (default, free, works everywhere)
        logger.info("Using Sentence-Transformers embeddings (local)")
        from .embeddings import SentenceTransformerEmbeddingClient
        return SentenceTransformerEmbeddingClient()


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    embedding_backend: str = None  # Will use env var if None
    groq_api_key: str = None
    
    def __post_init__(self):
        """Set embedding_backend from env if not provided."""
        if self.embedding_backend is None:
            self.embedding_backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    
    Workflow:
        1. Initialize: Create components
        2. Ingest: Chunk and embed documents
        3. Query: Retrieve and answer
    """
    def __init__(
        self,
        config: RAGConfig = None,
        embeddings=None,
        llm=None
    ):
        """
        Initialize RAG pipeline with all components.
        
        Args:
            config: RAGConfig object with settings
            embeddings: Optional embeddings client (for dependency injection)
            llm: Optional LLM client (for dependency injection)
        """
        load_env()
        self.config = config or RAGConfig()
        logger.info("Initializing RAG Pipeline...")

        # Use provided embeddings or create from config
        if embeddings:
            self.embeddings = embeddings
            logger.info("✓ Using provided embeddings client")
        else:
            try:
                self.embeddings = get_embeddings_client()
                logger.info("✓ Embeddings client ready")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                raise

        # Use provided LLM or create from config
        if llm:
            self.llm = llm
            logger.info("✓ Using provided LLM client")
        else:
            try:
                api_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GROQ_API_KEY not provided. Pass it in RAGConfig or set GROQ_API_KEY environment variable."
                    )
                self.llm = GroqLLMClient(api_key=api_key)
                logger.info("✓ LLM client ready")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise

        self.vector_store = ChromaVectorStore()
        logger.info("✓ Vector store ready")

        logger.info("✓ RAG Pipeline initialized")


    def ingest_pdf(
        self,
        pdf_path: str
    ) -> Dict[str, Any]:
        """
        Ingest a PDF file: extract text, chunk, and embed.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Ingestion stats
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> result = pipeline.ingest_pdf("research_paper.pdf")
            >>> print(f"Ingested {result['chunks_embedded']} chunks")
        """
        # Extract PDF
        processor = PDFProcessor(use_pdfplumber=False)
        text, metadata = processor.process_pdf(pdf_path)
        
        # Use filename (without extension) as doc_id
        doc_id = Path(pdf_path).stem
        
        # Add PDF metadata to chunks
        ingestion_result = self.ingest(doc_id, text)
        ingestion_result["pdf_metadata"] = metadata
        
        return ingestion_result
    
    def ingest_folder(
        self,
        folder_path: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Ingest all PDFs from a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
        
        Returns:
            Dict of {doc_id: ingestion_result}
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> results = pipeline.ingest_folder("./papers")
            >>> for doc_id, result in results.items():
            ...     print(f"{doc_id}: {result['chunks_embedded']} chunks")
        """
        processor = PDFProcessor(use_pdfplumber=False)
        documents = processor.process_folder(folder_path)
        
        results = {}
        for doc_id, (text, metadata) in documents.items():
            result = self.ingest(doc_id, text)
            result["pdf_metadata"] = metadata
            results[doc_id] = result
        
        return results

    def ingest(
        self,
        doc_id: str,
        text: str
    ) -> Dict[str, Any]:
        """
        Ingest a document: chunk it and embed each chunk.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
        
        Returns:
            Ingestion stats (chunks created, time taken, etc.)
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> result = pipeline.ingest(
            ...     "doc1",
            ...     "Machine learning is AI. Deep learning uses networks."
            ... )
            >>> print(f"Ingested {result['chunks_created']} chunks")
        """
        logger.info(f"Ingesting document: {doc_id}")

        #Step 1: chunk it
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        logger.info(f"✓ Chunks created: {len(chunks)}")

        if not chunks:
            logger.warning("No chunks created. Document may be too short.")
            return {
                "doc_id": doc_id,
                "chunks_created": 0,
                "time_taken": 0,
                "error": "Document too short"
            }

        #Step 2: embed each chunk
        chunks_embedded = 0
        for chunk in chunks:
            try:
                chunk_id = f"{doc_id}_chunk_{chunk.chunk_id}"
                embedding = self.embeddings.embed(chunk.text)
                self.vector_store.add(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_num": chunk.chunk_id,
                        "word_count": chunk.word_count
                    }
                )
                chunks_embedded += 1
            except Exception as e:
                logger.error(f"Failed to embed chunk {chunk_id}: {e}")
                continue
        
        logger.info(f"✓ Embedded {chunks_embedded}/{len(chunks)} chunks")
        return {
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "chunks_embedded": chunks_embedded,
            "status": "success" if chunks_embedded > 0 else "partial"
        }

    def query(
        self,
        query: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system: retrieve relevant chunks and generate answer.
        
        Args:
            query: User's question
            return_sources: Include source chunks in response
        
        Returns:
            Dictionary with 'query', 'answer', 'sources', etc.
        
        Raises:
            ValueError: If vector store is empty
        
        Example:
            >>> pipeline = RAGPipeline()
            >>> pipeline.ingest("doc1", "Machine learning is...")
            >>> result = pipeline.query("What is ML?")
            >>> print(result["answer"])
        """
        logger.info(f"Querying: {query}")

        #Check if we have docs
        if self.vector_store.size() == 0:
            raise ValueError("No documents in vector store")

        #Step 1: Embed the query
        query_embedding = self.embeddings.embed(query)
        logger.debug("  → Query embedded")
        
        #Step 2: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.retrieve(
            query_embedding,
            top_k=self.config.top_k
        )
        logger.debug(f"  → Retrieved {len(retrieved_chunks)} chunks")
        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "sources": [],
                "status": "no_results"
            }

        #Step 3: Build context from retrieved chunks
        context = build_context_string(retrieved_chunks)
        logger.debug(f"  → Built context ({len(context)} chars)")

        #Step 4: Query LLM with context
        try:
            answer = self.llm.query(context=context, query=query)
            logger.debug(f"  → LLM responded ({len(answer)} chars)")
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise

        #Step 5: Format response
        sources = [
            {
                "chunk_id": r.chunk_id,
                "similarity": round(r.similarity, 3),
                "preview": r.text[:100] + "..." if len(r.text) > 100 else r.text
            }
            for r in retrieved_chunks
        ] if return_sources else []
        
        result = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(retrieved_chunks),
            "status": "success"
        }
        
        logger.info(f"Query complete: {result['status']}")
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_chunks": self.vector_store.size(),
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "top_k": self.config.top_k
            }
        }