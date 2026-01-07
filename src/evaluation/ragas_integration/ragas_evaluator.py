"""
RAGAS Evaluator - Core evaluation logic using RAGAS framework
"""
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# RAGAS imports
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample

# LangChain for LLM wrapper (RAGAS requirement)
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RagasEvaluationResult:
    """Result from RAGAS evaluation."""
    eval_id: str
    query: str
    
    # RAGAS metrics (0-1 scale)
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    
    # Composite score
    ragas_score: float = 0.0
    
    # Metadata
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Calculate composite RAGAS score."""
        scores = [self.faithfulness, self.context_precision]
        valid_scores = [s for s in scores if s > 0]
        self.ragas_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0


class RagasEvaluator:
    """
    Evaluates RAG responses using RAGAS metrics.
    
    Metrics:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the retrieved chunks useful?
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            groq_api_key: Your Groq API key (or uses GROQ_API_KEY env var)
        """
        # TODO: Step 1 - Get API key
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY required")
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        self.evaluator_llm = LangchainLLMWrapper(llm)
        
        
        self.faithfulness = Faithfulness(llm=self.evaluator_llm)
        # self.answer_relevancy = ResponseRelevancy(llm=self.evaluator_llm)
        self.context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        
        # Storage for results
        self.results: List[RagasEvaluationResult] = []
        
        logger.info("✓ RAGAS Evaluator initialized (Faithfulness + Context Precision)")
    
    async def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RagasEvaluationResult:
        """
        Evaluate a single RAG response.
        """
        import time
        import hashlib
    
        start_time = time.time()  
        
        # 1. Create SingleTurnSample
        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth or ""
        )

        # 2. Score with each metric (async!)
        faithfulness_score = await self.faithfulness.single_turn_ascore(sample)
        # answer_relevancy_score = await self.answer_relevancy.single_turn_ascore(sample)
        answer_relevancy_score = None
        context_precision_score = await self.context_precision.single_turn_ascore(sample)
        
        # 3. Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # 4. Generate eval_id
        eval_id = hashlib.md5(f"{query}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # 5. Create and store result
        result = RagasEvaluationResult(
            eval_id=eval_id,
            query=query,
            faithfulness=float(faithfulness_score),
            answer_relevancy=0.0, #float(answer_relevancy_score),      
            context_precision=float(context_precision_score),    
            latency_ms=latency_ms
        )
        
        self.results.append(result)
        
        logger.info(f"Evaluation complete: RAGAS score = {result.ragas_score:.3f}")
        
        return result