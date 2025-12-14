---
title: Document Intelligence RAG
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "1.0"
app_file: src/main.py
pinned: false

---

# Document Intelligence RAG System

Production-grade Retrieval-Augmented Generation (RAG) system for analyzing research papers and documents with AI.
Ask questions about your PDFs. Get answers grounded in your documents with source attribution.

## Features

- PDF Ingestion: Extract text from PDFs using PDFProcessor
- Document Chunking: Split documents into smaller chunks for better context
- Embedding: Convert text chunks into vector embeddings using Ollama
- Vector Storage: Store embeddings in ChromaDB for efficient retrieval
- LLM Integration: Use Groq LLM for generating answers
- Source Attribution: Track document origins for citation
- FastAPI Integration: Build a REST API for easy access
- Docker Support: Containerize the system for easy deployment
- PDF Processing: Extract text from PDFs using PDFProcessor
- Document Chunking: Split documents into smaller chunks for better context
- Embedding: Convert text chunks into vector embeddings using Ollama
- Vector Storage: Store embeddings in ChromaDB for efficient retrieval
- LLM Integration: Use Groq LLM for generating answers
- Source Attribution: Track document origins for citation
- FastAPI Integration: Build a REST API for easy access
- Docker Support: Containerize the system for easy deployment

## Quickstart

### Prerequisites

- Python 3.12
- Ollama
- Groq API Key
- ChromaDB
- FastAPI
- Uvicorn
- PDFProcessor
- Embeddings
- LLM
- Vector Store

1. Setup environment variables
```bash
# Clone repository
git clone https://github.com/aankitdas/document-intelligence-rag.git
cd document-intelligence-rag

# Install Ollama (one-time setup)
# Download from https://ollama.ai
ollama pull nomic-embed-text

# Start Ollama server (in background)
ollama serve

# Create Python environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Set API keys
export GROQ_API_KEY="gsk_..."  # Get from https://console.groq.com
```

2. Prepare Documents
```bash
# Create a folder for documents
# Create papers folder
mkdir papers

# Add your PDFs to papers/
# Example: papers/research_paper.pdf
```
3. Run API
```bash
# Run API
uvicorn src.api.main:app --reload
```
4. Query API
```bash
# Query API
curl http://localhost:8000/ask -X POST -H "Content-Type: application/json" -d '{"question": "What is the main contribution of this paper?"}'
```

## Tech Stack

| Component        | Technology                    | Why                                                                 |
|------------------|-------------------------------|---------------------------------------------------------------------|
| Embeddings       | Ollama (`nomic-embed-text`)   | Local, free, 384-dimensional embeddings                             |
| Vector Database  | Chroma                        | Persistent storage, fast similarity search, completely free         |
| LLM              | Groq (Llama 3.1)              | Free API tier, very fast inference                                  |
| Backend          | FastAPI                       | Production-grade, async, automatic API docs                         |
| Frontend         | HTML / CSS / JavaScript       | Simple setup, no build tooling required                             |
| Package Manager  | UV                            | Fast dependency resolution, deterministic environments              |


#### Testing Github Actions sync to HF spaces 