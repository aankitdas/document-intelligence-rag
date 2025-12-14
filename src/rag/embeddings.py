"""
Embeddings module
----------------
Purpose: Convert text to vector embeddings using local Ollama or Sentence-Transformers
"""
import requests
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """
    Client for Ollama embedding service
    
    Requires: ollama serve running on localhost:11434
    Model: nomic-embed-text (384 dimensions)
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: int = 30
    ):

        """ 
        Initialize the Ollama embedding client
        Args:
            base_url: Ollama server URL
            model: Embedding model name
            timeout: Request timeout in seconds
        """

        self.base_url = base_url
        self.model = model
        self.timeout = timeout

        self._test_connection()

    def _test_connection(self) -> None:
        """Test if Ollama is running."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned {response.status_code}")
            
            logger.info(f"✓ Connected to Ollama at {self.base_url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Start it with: ollama serve"
            )

    def embed(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        Args:
            text: Text to embed
        
        Returns:
            List of floats (384 dimensions for nomic-embed-text)
        
        Raises:
            requests.RequestException: If Ollama API fails
        
        Example:
            >>> client = OllamaEmbeddingClient()
            >>> embedding = client.embed("Hello world")
            >>> len(embedding)
            384
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama error {response.status_code}: {response.text}"
                )
            
            # Extract embedding from response
            embedding = response.json()["embeddings"][0]
            return embedding
        
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Lost connection to Ollama at {self.base_url}"
            )
        except KeyError as e:
            raise ValueError(f"Unexpected Ollama response format: {e}")


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embeddings (one per text)
        
        Note: This calls Ollama for each text. For production,
              consider batching at the Ollama level.
        """
        embeddings = []
        for text in texts:
            try:
                emb = self.embed(text)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                raise
        
        return embeddings


class SentenceTransformerEmbeddingClient:
    """
    Client for Sentence-Transformers embeddings (local, free).
    
    No external service required - runs locally.
    Model: all-MiniLM-L6-v2 (384 dimensions)
    
    Install with: pip install sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize Sentence-Transformers embedding client.
        
        Args:
            model_name: HuggingFace model name
                Default: all-MiniLM-L6-v2 (fast, lightweight, 384 dims)
        
        Note: First initialization downloads the model (~500MB)
        """
        logger.info(f"Initializing Sentence-Transformers (model: {model_name})")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"✓ Loaded Sentence-Transformer model: {model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load Sentence-Transformer model: {e}")
            raise
    
    def embed(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats (384 dimensions for all-MiniLM-L6-v2)
        
        Example:
            >>> client = SentenceTransformerEmbeddingClient()
            >>> embedding = client.embed("Hello world")
            >>> len(embedding)
            384
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts (more efficient than calling embed() for each).
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embeddings (one per text)
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
    
    Returns:
        Similarity score from -1 to 1 (1 = identical)
    
    Note: Works best on normalized vectors (which both Ollama and Sentence-Transformers provide)
    
    Example:
        >>> vec1 = [1.0, 0.0, 0.0]
        >>> vec2 = [1.0, 0.0, 0.0]
        >>> cosine_similarity(vec1, vec2)
        1.0
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


# ============ TESTS ============

def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Identical vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity(vec1, vec2) - 1.0) < 0.01
    
    # Orthogonal vectors
    vec3 = [1.0, 0.0, 0.0]
    vec4 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec3, vec4) - 0.0) < 0.01


def test_cosine_similarity_normalized():
    """Test with normalized vectors."""
    # Normalized vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec1 = vec1 / np.linalg.norm(vec1)
    
    vec2 = np.array([1.0, 0.0, 0.0])
    vec2 = vec2 / np.linalg.norm(vec2)
    
    sim = cosine_similarity(vec1.tolist(), vec2.tolist())
    assert abs(sim - 1.0) < 0.01


if __name__ == "__main__":
    import os
    
    # Test based on EMBEDDING_BACKEND env var
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()
    
    try:
        if backend == "ollama":
            print("Testing Ollama embeddings...")
            client = OllamaEmbeddingClient()
        else:
            print("Testing Sentence-Transformers embeddings...")
            client = SentenceTransformerEmbeddingClient()
        
        # Test single embedding
        text = "Machine learning is AI"
        embedding = client.embed(text)
        
        print(f"✓ Embedding created: {len(embedding)} dimensions")
        print(f"  Sample values: {embedding[:5]}")
        
        # Test similarity
        text2 = "Deep learning uses networks"
        embedding2 = client.embed(text2)
        
        sim = cosine_similarity(embedding, embedding2)
        print(f"  Similarity between texts: {sim:.3f}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        if backend == "ollama":
            print("  Start Ollama with: ollama serve")
        else:
            print("  Install sentence-transformers with: pip install sentence-transformers")