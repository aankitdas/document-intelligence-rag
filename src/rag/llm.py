"""
LLM Module
----------
Purpose: Query Groq LLM with context for RAG answers
"""
from groq import Groq
from typing import List
import os
import logging 
logging.basicConfig(level=logging.INFO) 
from dotenv import load_dotenv

env_paths = [
    os.path.join(os.path.dirname(__file__), '../..', '.env'),  # Project root
    os.path.join(os.path.dirname(__file__), '.env'),  # Script directory
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
        break

logger = logging.getLogger(__name__)


class GroqLLMClient:
    """
    Client for querying Groq LLM with context for RAG answers
    Requires: Groq API key
    Model: llama-3.1-8b-instant -> check available models using client.models.list()
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.1-8b-instant",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        """
        Initialize Groq LLM client
        Args:
            api_key (str): Groq API key
            model_name (str): Groq model name
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): 0-1, higher for more creative shit
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        logger.info(f"Groq LLM client initialized with model: {self.model_name}")

    def _build_prompt(
        self,
        context: str,
        question: str,
    ) -> str:
        """
        Build the final prompt for LLM
        Args:
            context (str): Retrieved chunks
            question (str): Question to ask
        Returns:
            str: Prompt for LLM
        """
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
                    If the context doesn't contain enough information to answer, say so explicitly.
                    Do not make up information.

                    Context: {context}

                    Question: {question}

                    Answer:"""
        return prompt

    def query(
        self,
        context: str,
        query: str,
    ) -> str:
        """
        Query the Groq LLM with context
        Args:
            context (str): Retrieved context from vector store
            query: User's question
        
        Returns:
            LLM's answer as string
        
        Raises:
            RuntimeError: If Groq API fails
        """
        try:
            prompt = self._build_prompt(context, query)
            logger.debug(f"Querying Groq with {len(context)} chars context")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            answer = response.choices[0].message.content
            logger.debug(f"Groq API response: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Groq query failed: {e}")
            raise RuntimeError(f"LLM query failed: {e}")

    def query_with_sources(
        self,
        context: str,
        query: str,
        sources: List[str] = None
    ) -> dict:
        """
        Query LLM and return answer with source attribution.
        
        Args:
            context: Retrieved context
            query: User's question
            sources: Optional list of source identifiers (chunk IDs, URLs, etc.)
        
        Returns:
            Dict with 'answer' and 'sources' keys
        
        Example:
            >>> result = client.query_with_sources(
            ...     context="...",
            ...     query="What is ML?",
            ...     sources=["doc1_chunk_0", "doc1_chunk_2"]
            ... )
            >>> print(result["answer"])
            >>> print(result["sources"])
        """
        answer = self.query(context, query)
        
        return {
            "answer": answer,
            "sources": sources or []
        }

def build_context_string(
    retrieved_results: List,
    include_scores: bool = True
) -> str:
    """
    Build a context string from retrieved results
    Args:
        retrieved_results: List of retrieved results
        include_scores: Whether to include scores in the context string
    Returns:
        Context string
    """
    context_parts = []

    for i, result in enumerate(retrieved_results, 1):
        if include_scores:
            part = f"[Chunk {i} - Relevance: {result.similarity:.1%}]\n{result.text}"
        else:
            part = f"[Chunk {i}]\n{result.text}"
        
        context_parts.append(part)

    return "\n\n".join(context_parts)
            
# ============ TESTS ============

def test_build_context_string():
    """Test context string building."""
    from .vector_store import RetrievalResult
    
    results = [
        RetrievalResult("chunk1", "Text 1", 0.95),
        RetrievalResult("chunk2", "Text 2", 0.87)
    ]
    
    context = build_context_string(results)
    
    assert "Text 1" in context
    assert "Text 2" in context
    assert "95.0%" in context


if __name__ == "__main__":
    try:
        # Test Groq client
        client = GroqLLMClient(api_key=os.getenv("GROQ_API_KEY"))
        
        # Test context string
        from .vector_store import RetrievalResult
        
        results = [
            RetrievalResult("chunk1", "Machine learning is AI", 0.95),
            RetrievalResult("chunk2", "Deep learning uses neural networks", 0.87)
        ]
        
        context = build_context_string(results)
        
        # Query
        answer = client.query(
            context=context,
            query="What is machine learning?"
        )
        
        print("✓ Groq query successful!")
        print(f"Answer: {answer[:200]}...")
    
    except Exception as e:
        print(f"✗ Error: {e}")