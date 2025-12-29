"""
Sample script to generate evaluation results for testing/demo purposes.
Run this to populate the evaluation dashboard with realistic data.

Usage:
    python sample_evaluation_data.py
"""
import os
import random
import numpy as np
from src.evaluation import RAGEvaluator, EvaluationResult

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")

# Sample medical/pharma queries for realistic context
SAMPLE_QUERIES = [
    "What are the primary side effects of this drug?",
    "What is the mechanism of action for this treatment?",
    "What were the patient demographics in the clinical trial?",
    "What is the recommended dosage for this medication?",
    "What are the contraindications for this therapy?",
    "What is the success rate from the phase II trial?",
    "How does this drug compare to existing treatments?",
    "What are the inclusion/exclusion criteria for this study?",
    "What is the safety profile based on reported adverse events?",
    "What biomarkers should be monitored during treatment?",
]

SAMPLE_DOCS = [
    "FDA_Approval_Summary.pdf",
    "Clinical_Trial_Protocol.pdf",
    "Safety_Profile_Report.pdf",
    "Pharmacokinetics_Study.pdf",
    "Adverse_Events_Listing.pdf",
]

def generate_realistic_metrics(quality_level: float = 0.85) -> dict:
    """
    Generate realistic evaluation metrics.
    quality_level: 0.0-1.0, controls how good the metrics are
    """
    noise = random.gauss(0, 0.05)  # Add some natural variation
    quality = np.clip(quality_level + noise, 0.0, 1.0)
    
    return {
        "retrieval_precision": np.clip(quality + random.gauss(0, 0.08), 0.6, 1.0),
        "retrieval_recall": np.clip(quality + random.gauss(0, 0.1), 0.5, 1.0),
        "rank_position": random.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0],
        "rouge_l": np.clip(quality - 0.1 + random.gauss(0, 0.08), 0.4, 0.95),
        "bert_score": np.clip(quality + random.gauss(0, 0.05), 0.65, 0.99),
        "answer_relevance": np.clip(quality - 0.05 + random.gauss(0, 0.06), 0.6, 0.98),
        "faithfulness": np.clip(quality + random.gauss(0, 0.04), 0.7, 0.99),
        "hallucination_detected": random.random() > (quality * 1.2),  # Better quality = fewer hallucinations
        "source_attribution_score": np.clip(quality - 0.05 + random.gauss(0, 0.07), 0.65, 0.99),
        "latency_ms": random.gauss(300, 100),  # Average 300ms with 100ms std dev
        "tokens_used": random.randint(80, 250),
        "cost_cents": random.uniform(0.15, 0.8),
    }

def generate_sample_results(num_queries: int = 30, cto_demo: bool = True):
    """
    Generate sample evaluation results and add to evaluator.
    
    Args:
        num_queries: Number of evaluation results to generate
        cto_demo: If True, skew results toward good performance (to impress CTO)
    """
    evaluator = RAGEvaluator(store_results=True, results_dir=EVAL_DIR)
    
    print(f"🔧 Generating {num_queries} sample evaluation results...")
    
    for i in range(num_queries):
        query = random.choice(SAMPLE_QUERIES)
        source_docs = random.sample(SAMPLE_DOCS, k=random.randint(1, 4))
        
        # If CTO demo mode, bias toward good metrics
        quality_level = 0.88 if cto_demo else random.uniform(0.6, 0.95)
        metrics = generate_realistic_metrics(quality_level)
        
        # Create realistic answer (shorter answers are often better)
        answer = f"Based on the clinical data, {query[:-1].lower()}. This finding is supported by the source documents indicating a positive correlation with treatment outcomes."
        
        result = EvaluationResult(
            query=query,
            answer=answer,
            source_docs=source_docs,
            num_retrieved=len(source_docs),
            retrieval_precision=metrics["retrieval_precision"],
            retrieval_recall=metrics["retrieval_recall"],
            rank_position=metrics["rank_position"],
            rouge_l=metrics["rouge_l"],
            bert_score=metrics["bert_score"],
            answer_relevance=metrics["answer_relevance"],
            faithfulness=metrics["faithfulness"],
            hallucination_detected=metrics["hallucination_detected"],
            source_attribution_score=metrics["source_attribution_score"],
            latency_ms=metrics["latency_ms"],
            tokens_used=metrics["tokens_used"],
            cost_cents=metrics["cost_cents"],
        )
        
        evaluator.add_result(result)
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ Generated {i + 1}/{num_queries} results")
    
    # Print summary
    metrics = evaluator.compute_aggregate_metrics()
    print(f"\n✅ Sample data generated! Summary:")
    print(f"  • Total evaluations: {metrics['total_evaluations']}")
    print(f"  • Avg Precision: {metrics['retrieval_precision_mean']:.3f}")
    print(f"  • Avg BERTScore: {metrics['bert_score_mean']:.3f}")
    print(f"  • Faithfulness: {metrics['faithfulness_mean']:.3f}")
    print(f"  • Hallucination Rate: {metrics['hallucination_rate']*100:.1f}%")
    print(f"  • Avg Latency: {metrics['latency_mean']:.0f}ms")
    print(f"  • Avg Cost: ${metrics['cost_per_query']/100:.4f}")
    print(f"\n🌐 View dashboard at: http://localhost:8000/evaluation")

def clear_previous_results():
    """Clear any existing results before generating new ones."""
    evaluator = RAGEvaluator(store_results=True, results_dir="evaluation_results")
    evaluator.reset()
    print("🗑️  Cleared previous results")

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("RAG Evaluation Sample Data Generator")
    print("=" * 60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear":
            clear_previous_results()
            sys.exit(0)
        elif sys.argv[1] == "--cto-demo":
            print("\n📊 Generating CTO demo dataset (high quality metrics)...\n")
            generate_sample_results(num_queries=50, cto_demo=True)
        elif sys.argv[1] == "--realistic":
            print("\n📊 Generating realistic mixed-quality dataset...\n")
            generate_sample_results(num_queries=50, cto_demo=False)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python sample_evaluation_data.py [--clear|--cto-demo|--realistic]")
            sys.exit(1)
    else:
        # Default: clear and generate CTO demo
        clear_previous_results()
        print()
        generate_sample_results(num_queries=30, cto_demo=True)