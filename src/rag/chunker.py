"""
Chunker module
--------------
Purpose: Split text into smaller chunks.
"""

from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    chunk_id: int
    start_idx: int
    word_count: int

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Split text into smaller chunks.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int): The size of each chunk.
        overlap (int): The overlap between chunks.

    Returns:
        List[Chunk]: A list of chunks.
    """
    words = text.split()
    
    if not words:
        return []

    stride = chunk_size - overlap
    chunks = []
    chunk_id = 0
    
    for i in range(0, len(words), stride):
        chunk = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk)
        
        if not chunk_text.strip():
            continue

        chunk = Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            start_idx=i,
            word_count=len(chunk)
        )
        
        chunks.append(chunk)
        chunk_id += 1
        
    return chunks


def chunk_documents(
    documents: Dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50
) -> Dict[str, List[Chunk]]:
    """
    Chunk multiple documents.
    
    Args:
        documents: Dict of {doc_id: text}
        chunk_size: Tokens per chunk
        overlap: Token overlap
    
    Returns:
        Dict of {doc_id: [chunks]}
    
    Example:
        >>> docs = {"doc1": "Text 1", "doc2": "Text 2"}
        >>> chunked = chunk_documents(docs)
        >>> "doc1" in chunked
        True
    """
    chunked_docs = {}
    
    for doc_id, text in documents.items():
        chunks = chunk_text(text, chunk_size, overlap)
        chunked_docs[doc_id] = chunks
    
    return chunked_docs

if __name__ == "__main__":
    text = """
    Machine Learning is a subset of artificial intelligence that involves training models to make predictions or decisions based on data. It is a powerful tool for solving a wide range of problems, from image recognition to natural language processing. In this article, we will explore the basics of machine learning and how it can be used to solve real-world problems.
    """
    
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    print(f"Split into {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  Chunk {chunk.chunk_id}: {chunk.word_count} words | {chunk.text[:60]}...")

