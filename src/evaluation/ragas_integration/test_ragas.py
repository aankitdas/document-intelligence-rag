"""
Quick test for RAGAS integration
Run: python -m src.evaluation.ragas_integration.test_ragas
"""
import asyncio
from src.rag import RAGPipeline, RAGConfig
from src.evaluation.ragas_integration import RagasReadyPipeline, RagasEvaluator


async def test_ragas():
    print("=" * 50)
    print("Testing RAGAS Integration")
    print("=" * 50)
    
    # Step 1: Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = RAGPipeline(RAGConfig())
    ragas_pipeline = RagasReadyPipeline(pipeline)
    print("   ✓ Pipeline ready")
    
    # Step 2: Initialize evaluator
    print("\n2. Initializing RAGAS evaluator...")
    evaluator = RagasEvaluator()
    print("   ✓ Evaluator ready")
    
    # Step 3: Ingest a test document
    print("\n3. Ingesting test document...")
    test_text = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn from data. Deep learning is a type of machine learning 
    that uses neural networks with multiple layers. Natural language processing 
    (NLP) is used to understand human language.
    """
    pipeline.ingest("test_doc", test_text)
    print(f"   ✓ Ingested {pipeline.vector_store.size()} chunks")
    
    # Step 4: Query with full context capture
    print("\n4. Querying pipeline...")
    response = ragas_pipeline.query_for_evaluation("What is machine learning?")
    print(f"   Query: {response.query}")
    print(f"   Answer: {response.answer[:100]}...")
    print(f"   Contexts captured: {len(response.contexts)}")
    print(f"   Status: {response.status}")
    
    # Step 5: Evaluate with RAGAS
    print("\n5. Running RAGAS evaluation...")
    result = await evaluator.evaluate_single(
        query=response.query,
        answer=response.answer,
        contexts=response.contexts
    )
    
    print(f"\n{'=' * 50}")
    print("RAGAS RESULTS")
    print(f"{'=' * 50}")
    print(f"   Faithfulness:      {result.faithfulness:.3f}")
    print(f"   Answer Relevancy:  {result.answer_relevancy:.3f}")
    print(f"   Context Precision: {result.context_precision:.3f}")
    print(f"   ─────────────────────────────")
    print(f"   RAGAS Score:       {result.ragas_score:.3f}")
    print(f"   Eval Time:         {result.latency_ms:.0f}ms")
    print(f"{'=' * 50}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_ragas())