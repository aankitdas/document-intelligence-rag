"""
Pipeline Adapter - Captures full context for RAGAS evaluation
"""
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.rag.llm import build_context_string


@dataclass
class EvaluationReadyResponse:
    """Response with full contexts for RAGAS."""
    query: str
    answer: str
    contexts: List[str]  # Full text of each retrieved chunk
    sources: List[Dict]  # Original source metadata
    chunks_used: int
    response_time_ms: float
    status: str


class RagasReadyPipeline:
    """
    Wraps your RAGPipeline to capture full context.
    """
    
    def __init__(self, base_pipeline):
        """
        Args:
            base_pipeline: Your existing RAGPipeline instance
        """
        self.pipeline = base_pipeline
        self.config = base_pipeline.config
        self.embeddings = base_pipeline.embeddings
        self.llm = base_pipeline.llm
        self.vector_store = base_pipeline.vector_store
    
    def query_for_evaluation(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> EvaluationReadyResponse:
        """
        Query and capture FULL context for RAGAS.
        
        TODO: Implement this by:
        1. Embedding the query (use self.pipeline.embeddings)
        2. Retrieving chunks (use self.pipeline.vector_store.retrieve)
        3. Extracting FULL text from each chunk
        4. Building context for LLM
        5. Getting answer from LLM
        6. Returning EvaluationReadyResponse with full contexts
        """
        start_time = time.time()
        
        if self.vector_store.size() == 0:
            return EvaluationReadyResponse(
                query=query,
                answer="No documents in vector store",
                contexts=[],
                sources=[],
                chunks_used=0,
                response_time_ms=0,
                status="no_documents"
            )

        query_embedding = self.embeddings.embed(query)
        k = top_k or self.config.top_k
        retrieved_chunks = self.vector_store.retrieve(query_embedding, top_k=k)
        if not retrieved_chunks:
            return EvaluationReadyResponse(
                query=query,
                answer="No relevant documents found.",
                contexts=[],
                sources=[],
                chunks_used=0,
                response_time_ms=(time.time() - start_time) * 1000,
                status="no_results"
            )
        contexts = [chunk.text for chunk in retrieved_chunks]

        context_string = build_context_string(retrieved_chunks)
        answer = self.llm.query(context=context_string, query=query)
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "similarity": round(chunk.similarity, 3),
                "preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            }
            for chunk in retrieved_chunks
        ]

        response_time_ms = (time.time() - start_time) * 1000

        return EvaluationReadyResponse(
            query=query,
            answer=answer,
            contexts=contexts,  # Full texts for RAGAS!
            sources=sources,
            chunks_used=len(contexts),
            response_time_ms=response_time_ms,
            status="success"
        )