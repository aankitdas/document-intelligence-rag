"""
RAGAS API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# We'll set these from main.py
ragas_pipeline = None
ragas_evaluator = None


class RagasEvalRequest(BaseModel):
    """Direct evaluation request."""
    query: str
    answer: str
    contexts: List[str]


class RagasQueryRequest(BaseModel):
    """Query + evaluate request."""
    query: str
    top_k: int = 3


def init_ragas_router(pipeline, evaluator):
    """Initialize router with pipeline and evaluator instances."""
    global ragas_pipeline, ragas_evaluator
    ragas_pipeline = pipeline
    ragas_evaluator = evaluator
    return router


@router.post("/evaluate")
async def evaluate_direct(request: RagasEvalRequest):
    """
    Evaluate a query-answer pair directly.
    
    Use this when you already have the answer and contexts.
    """
    if not ragas_evaluator:
        raise HTTPException(status_code=503, detail="RAGAS not initialized")
    
    result = await ragas_evaluator.evaluate_single(
        query=request.query,
        answer=request.answer,
        contexts=request.contexts
    )
    
    return {
        "eval_id": result.eval_id,
        "faithfulness": result.faithfulness,
        "context_precision": result.context_precision,
        "ragas_score": result.ragas_score,
        "latency_ms": round(result.latency_ms, 2)
    }


@router.post("/query-and-evaluate")
async def query_and_evaluate(request: RagasQueryRequest):
    """
    Query the RAG system AND evaluate the response.
    
    Returns both the answer and RAGAS metrics.
    """
    if not ragas_pipeline or not ragas_evaluator:
        raise HTTPException(status_code=503, detail="RAGAS not initialized")
    
    # Step 1: Query pipeline
    response = ragas_pipeline.query_for_evaluation(
        query=request.query,
        top_k=request.top_k
    )
    
    if response.status != "success":
        return {
            "query": response.query,
            "answer": response.answer,
            "status": response.status,
            "ragas": None
        }
    
    # Step 2: Evaluate with RAGAS
    eval_result = await ragas_evaluator.evaluate_single(
        query=response.query,
        answer=response.answer,
        contexts=response.contexts
    )
    
    return {
        "query": response.query,
        "answer": response.answer,
        "sources": response.sources,
        "chunks_used": response.chunks_used,
        "response_time_ms": round(response.response_time_ms, 2),
        "ragas": {
            "eval_id": eval_result.eval_id,
            "faithfulness": eval_result.faithfulness,
            "context_precision": eval_result.context_precision,
            "ragas_score": eval_result.ragas_score,
            "eval_time_ms": round(eval_result.latency_ms, 2)
        }
    }


@router.get("/metrics")
async def get_metrics():
    """Get aggregate RAGAS metrics from all evaluations."""
    if not ragas_evaluator:
        raise HTTPException(status_code=503, detail="RAGAS not initialized")
    
    results = ragas_evaluator.results
    
    if not results:
        return {"total_evaluations": 0, "message": "No evaluations yet"}
    
    # Calculate averages
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_precision = sum(r.context_precision for r in results) / len(results)
    avg_ragas = sum(r.ragas_score for r in results) / len(results)
    
    return {
        "total_evaluations": len(results),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_context_precision": round(avg_precision, 3),
        "avg_ragas_score": round(avg_ragas, 3)
    }