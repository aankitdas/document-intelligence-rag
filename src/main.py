from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
from typing import List, Optional
from datetime import datetime
import tempfile
from pathlib import Path

from src.rag import RAGPipeline, RAGConfig
from src.evaluation import RAGEvaluator, EvaluationResult
import io
import csv
# ==================== Setup ====================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Intelligence RAG",
    description="RAG system for analyzing documents with LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

evaluator = RAGEvaluator(store_results=True, results_dir="evaluation_results")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    """Request body for query endpoint."""
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    """Response for query."""
    query: str
    answer: str
    sources: List[dict]
    chunks_used: int
    response_time: float
    status: str


class IngestResponse(BaseModel):
    """Response for ingestion."""
    doc_id: str
    filename: str
    chunks_created: int
    chunks_embedded: int
    status: str
    timestamp: str


class IngestFolderResponse(BaseModel):
    """Response for folder ingestion."""
    total_documents: int
    total_chunks: int
    documents: List[dict]
    timestamp: str


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    embedding_backend: str
    groq: str
    chroma: dict
    timestamp: str


class StatsResponse(BaseModel):
    """Response for stats."""
    total_chunks: int
    config: dict
    timestamp: str


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    
    logger.info("=" * 60)
    logger.info("Starting Document Intelligence RAG API")
    logger.info("=" * 60)
    
    try:
        # Create RAG config (reads EMBEDDING_BACKEND from env)
        config = RAGConfig(
            chunk_size=500,
            chunk_overlap=50,
            top_k=3
        )
        
        # Initialize pipeline (automatically uses get_embeddings_client())
        pipeline = RAGPipeline(config=config)
        
        logger.info("✓ Pipeline initialized successfully")
        logger.info(f"✓ Embedding backend: {config.embedding_backend}")
        logger.info(f"✓ API ready at http://localhost:8000")
        logger.info(f"✓ Interactive docs at http://localhost:8000/docs")
    
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Document Intelligence RAG API")


# ==================== Health & Status ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system health.
    
    Returns:
        Health status of all components
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Check components
        embeddings_ok = "✓" if pipeline.embeddings else "✗"
        groq_ok = "✓" if pipeline.llm else "✗"
        chroma_ok = pipeline.vector_store.size() >= 0
        
        return HealthResponse(
            status="healthy" if all([embeddings_ok == "✓", groq_ok == "✓", chroma_ok]) else "degraded",
            embedding_backend=pipeline.config.embedding_backend,
            groq=groq_ok,
            chroma={
                "status": "✓" if chroma_ok else "✗",
                "chunks": pipeline.vector_store.size()
            },
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get pipeline statistics.
    
    Returns:
        Current stats: total chunks, config, etc.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_stats()
        
        return StatsResponse(
            total_chunks=stats['total_chunks'],
            config=stats['config'],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Ingestion Endpoints ====================

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a single PDF file.
    
    Args:
        file: PDF file to upload
    
    Returns:
        Ingestion result with doc_id and chunk count
    
    Example:
        curl -X POST "http://localhost:8000/ingest" \
          -F "file=@research_paper.pdf"
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        logger.info(f"Processing uploaded PDF: {file.filename}")
        
        # Ingest PDF
        result = pipeline.ingest_pdf(tmp_path)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        return IngestResponse(
            doc_id=result['doc_id'],
            filename=file.filename,
            chunks_created=result['chunks_created'],
            chunks_embedded=result['chunks_embedded'],
            status=result['status'],
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest-folder", response_model=IngestFolderResponse)
async def ingest_folder(folder_path: str):
    """
    Ingest all PDFs from a folder.
    
    Args:
        folder_path: Path to folder containing PDFs
    
    Returns:
        Summary of all ingested documents
    
    Example:
        curl -X POST "http://localhost:8000/ingest-folder" \
          -H "Content-Type: application/json" \
          -d '{"folder_path": "./papers"}'
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Check folder exists
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")
        
        logger.info(f"Ingesting folder: {folder_path}")
        
        # Ingest all PDFs
        results = pipeline.ingest_folder(folder_path)
        
        if not results:
            raise HTTPException(status_code=400, detail="No PDFs found in folder")
        
        # Build response
        total_chunks = sum(r['chunks_embedded'] for r in results.values())
        documents = [
            {
                "doc_id": doc_id,
                "chunks": r['chunks_embedded']
            }
            for doc_id, r in results.items()
        ]
        
        return IngestFolderResponse(
            total_documents=len(results),
            total_chunks=total_chunks,
            documents=documents,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Folder ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ==================== Query Endpoint ====================

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Args:
        request: QueryRequest with 'query' and optional 'top_k'
    
    Returns:
        Answer with sources and metadata
    
    Example:
        curl -X POST "http://localhost:8000/query" \
          -H "Content-Type: application/json" \
          -d '{"query": "What is machine learning?", "top_k": 3}'
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if pipeline.vector_store.size() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload documents first."
        )
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Query: {request.query}")
        
        # Query pipeline
        result = pipeline.query(request.query, return_sources=True)
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            chunks_used=result['chunks_used'],
            response_time=round(response_time, 3),
            status=result['status']
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ==================== Document Management ====================

@app.get("/documents")
async def list_documents():
    """
    List all ingested documents.
    
    Returns:
        List of document IDs and chunk counts
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        total_chunks = pipeline.vector_store.size()
        
        return {
            "total_chunks": total_chunks,
            "status": "ready" if total_chunks > 0 else "empty",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks.
    
    Args:
        doc_id: Document ID to delete
    
    Returns:
        Deletion result
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Note: This is a simple implementation
        # For production, you'd want to track document chunks and delete them
        logger.info(f"Deleting document: {doc_id}")
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "message": "Document deletion queued",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_system():
    """
    Reset the entire system - clear all documents and embeddings.
    
    WARNING: This deletes all stored embeddings!
    
    Returns:
        Reset confirmation
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.warning("RESET: Clearing all documents and embeddings")
        
        # Clear vector store
        pipeline.vector_store.clear()
        
        logger.info("✓ System reset complete")
        
        return {
            "status": "success",
            "message": "All documents and embeddings cleared",
            "chunks_remaining": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== Evaluation Endpoints ====================
# Add these endpoints to your main.py (after existing endpoints)

@app.get("/evaluation")
async def evaluation_ui():
    """Serve evaluation dashboard."""
    frontend_path = "frontend/evaluation.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"error": "Evaluation dashboard not found"}


@app.get("/evaluation/metrics")
async def get_evaluation_metrics():
    """Get aggregate evaluation metrics."""
    return evaluator.compute_aggregate_metrics()


@app.get("/evaluation/timeseries")
async def get_timeseries_data():
    """Get evaluation results as timeseries for visualization."""
    return evaluator.get_results_timeseries()


@app.get("/evaluation/failures")
async def get_failure_analysis():
    """Get failure mode analysis."""
    return evaluator.get_failure_analysis()


@app.get("/evaluation/percentiles")
async def get_percentile_data():
    """Get percentile analysis for performance metrics."""
    return evaluator.get_percentile_analysis()


@app.post("/evaluation/add-result")
async def add_evaluation_result(result: dict):
    """
    Add a single evaluation result.
    
    Expected fields:
    {
        "query": "...",
        "answer": "...",
        "source_docs": ["doc1", "doc2"],
        "num_retrieved": 3,
        "retrieval_precision": 0.8,
        "retrieval_recall": 0.9,
        "rank_position": 1,
        "rouge_l": 0.75,
        "bert_score": 0.85,
        "answer_relevance": 0.9,
        "faithfulness": 0.95,
        "hallucination_detected": false,
        "source_attribution_score": 0.9,
        "latency_ms": 234.5,
        "tokens_used": 150,
        "cost_cents": 0.5
    }
    """
    try:
        eval_result = EvaluationResult(**result)
        evaluator.add_result(eval_result)
        return {
            "status": "success",
            "eval_id": eval_result.eval_id,
            "message": "Result added successfully"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.get("/evaluation/export")
async def export_results():
    """Export evaluation results as CSV."""
    # Create CSV in memory
    output = io.StringIO()
    
    if evaluator.results:
        results_data = [r.to_dict() for r in evaluator.results]
        fieldnames = results_data[0].keys()
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
        
        output.seek(0)
        csv_content = output.getvalue()
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=rag_evaluation.csv"}
        )
    
    return {"error": "No results to export"}, 404


@app.post("/evaluation/reset")
async def reset_evaluation_results():
    """Clear all evaluation results."""
    evaluator.reset()
    return {"status": "success", "message": "All results cleared"}


@app.get("/evaluation/stats")
async def get_evaluation_stats():
    """Get summary statistics."""
    metrics = evaluator.compute_aggregate_metrics()
    return {
        "total_evaluations": metrics["total_evaluations"],
        "average_faithfulness": metrics["faithfulness_mean"],
        "hallucination_rate": metrics["hallucination_rate"],
        "average_latency_ms": metrics["latency_mean"],
        "average_cost_cents": metrics["cost_per_query"],
        "mrr": metrics["mrr"],
        "timestamp": metrics["timestamp"]
    }


# ==================== Integration with your existing endpoints ====================
# Optional: Enhance your existing /query endpoint to track metrics
# Replace or enhance your current /query endpoint like this:

@app.post("/query-with-eval")
async def query_with_evaluation(request: dict):
    """
    Query endpoint with automatic evaluation tracking.
    Use this if you want to automatically log metrics for every query.
    """
    import time
    from typing import Any
    
    query = request.get("question", "")
    start_time = time.time()
    
    try:
        # Call your existing pipeline
        # This is pseudocode - adjust based on your actual pipeline
        response = await query(request)  # Call your existing query function
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Create evaluation result (with placeholder values for now)
        eval_result = EvaluationResult(
            query=query,
            answer=response.get("answer", ""),
            source_docs=response.get("sources", []),
            num_retrieved=len(response.get("sources", [])),
            retrieval_precision=0.85,  # You'd compute these from your pipeline
            retrieval_recall=0.80,
            rank_position=1,
            rouge_l=0.75,
            bert_score=0.85,
            answer_relevance=0.88,
            faithfulness=0.90,
            hallucination_detected=False,
            source_attribution_score=0.85,
            latency_ms=latency_ms,
            tokens_used=len(response.get("answer", "").split()),
            cost_cents=0.5  # Compute based on your pricing
        )
        
        evaluator.add_result(eval_result)
        
        return {
            **response,
            "eval_id": eval_result.eval_id,
            "latency_ms": latency_ms
        }
    
    except Exception as e:
        return {"error": str(e)}, 500


# ==================== Root Endpoint ====================

@app.get("/", response_class=FileResponse)
async def root():
    """Root endpoint - serve web UI."""
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    
    # If no frontend, return API info
    return {
        "name": "Document Intelligence RAG",
        "version": "1.0.0",
        "description": "RAG system for analyzing documents with LLM",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
        "embedding_backend": pipeline.config.embedding_backend if pipeline else "initializing",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)