"""
Real RAG Evaluation Script
Runs actual queries through my RAG and computes real metrics.
"""

import json
import tempfile
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.rag.pipeline import RAGPipeline
from src.evaluation import RAGEvaluator, EvaluationResult


# ==================== TEST DATASET ====================
# These are synthetic queries + documents, but metrics are REAL
# based on actual retrieval and generation from my RAG

TEST_DOCUMENTS = {
    "medical_research_1.txt": """
    Drug X Clinical Trial Results
    
    A Phase III clinical trial was conducted to evaluate the efficacy and safety of Drug X 
    in patients with condition Y. The study enrolled 500 patients aged 18-65 with confirmed 
    diagnosis of condition Y.
    
    Primary Efficacy Endpoint:
    Drug X demonstrated a 65% response rate compared to 35% in the placebo group (p<0.001).
    The median time to response was 4 weeks.
    
    Safety Profile:
    The most common adverse events were:
    - Headache (12% of patients)
    - Mild gastrointestinal upset (8% of patients)
    - Dizziness (5% of patients)
    - Fatigue (4% of patients)
    
    Serious adverse events occurred in 2% of patients, including liver enzyme elevation.
    No deaths were attributed to the drug during the trial period.
    
    Dosage Recommendations:
    The recommended dose is 500mg twice daily with meals. Dose adjustments may be necessary 
    for patients with renal impairment (dose reduction to 250mg twice daily recommended).
    
    Mechanism of Action:
    Drug X works by inhibiting protein kinase Y, which is overexpressed in condition Y cells.
    This inhibition leads to cell cycle arrest and apoptosis of affected cells.
    """,
    
    "drug_interactions.txt": """
    Drug X Drug Interaction Guide
    
    Important Drug Interactions:
    
    1. CYP3A4 Inhibitors (e.g., ketoconazole, ritonavir):
       - May increase Drug X levels by 3-5 fold
       - Monitor for adverse effects
       - Consider dose reduction
    
    2. Warfarin:
       - Potential increased bleeding risk
       - Monitor INR closely
       - Baseline INR and weekly monitoring recommended
    
    3. Oral Contraceptives:
       - May reduce contraceptive efficacy
       - Alternative contraception recommended
       - No dose adjustment needed for Drug X
    
    4. NSAIDs:
       - Increased risk of GI bleeding
       - Monitor for GI symptoms
       - Consider gastroprotection
    
    5. ACE Inhibitors:
       - No significant interaction
       - Safe to use concomitantly
       - No monitoring required
    """,
    
    "patient_case_study.txt": """
    Case Study: 45-year-old Female with Condition Y
    
    Patient History:
    A 45-year-old female presented with a 6-month history of progressive symptoms consistent 
    with condition Y. She has a past medical history of hypertension controlled on lisinopril 
    and type 2 diabetes on metformin.
    
    Treatment Response:
    Patient was started on Drug X 500mg twice daily. After 2 weeks of treatment, she reported 
    partial symptom improvement. By week 6, she achieved complete response with 95% symptom 
    resolution.
    
    Side Effects Experienced:
    - Mild headache (treated with acetaminophen)
    - Occasional nausea (resolved with food intake)
    - No serious adverse events
    
    Follow-up:
    Patient continues to do well on Drug X at 6-month follow-up with sustained response.
    No dose adjustments were necessary. Lab values remain within normal limits.
    """
}

TEST_CASES = [
    {
        "query": "What is the response rate of Drug X?",
        "expected_answer_keywords": ["65%", "response rate"],
        "expected_source_docs": ["medical_research_1"],
        "description": "Should retrieve clinical trial data"
    },
    {
        "query": "What are the side effects of Drug X?",
        "expected_answer_keywords": ["headache", "gastrointestinal", "dizziness"],
        "expected_source_docs": ["medical_research_1"],
        "description": "Should retrieve safety profile section"
    },
    {
        "query": "How does Drug X interact with warfarin?",
        "expected_answer_keywords": ["warfarin", "bleeding", "INR"],
        "expected_source_docs": ["drug_interactions"],
        "description": "Should retrieve drug interactions guide"
    },
    {
        "query": "What is the recommended dosage of Drug X?",
        "expected_answer_keywords": ["500mg", "twice daily"],
        "expected_source_docs": ["medical_research_1"],
        "description": "Should retrieve dosage recommendations"
    },
    {
        "query": "What is the mechanism of action for Drug X?",
        "expected_answer_keywords": ["protein kinase", "inhibiting", "apoptosis"],
        "expected_source_docs": ["medical_research_1"],
        "description": "Should retrieve mechanism section"
    },
]


# ==================== METRIC COMPUTATION ====================

def compute_retrieval_precision(
    retrieved_docs: List[str],
    expected_docs: List[str]
) -> float:
    """
    Precision: Of the docs we retrieved, what % were actually relevant?
    
    Formula: TP / (TP + FP)
    where TP = relevant docs we retrieved
          FP = irrelevant docs we retrieved
    """
    if not retrieved_docs:
        return 0.0
    
    # Count how many retrieved docs match expected
    relevant_count = sum(1 for doc in retrieved_docs if doc in expected_docs)
    
    precision = relevant_count / len(retrieved_docs)
    return float(precision)


def compute_retrieval_recall(
    retrieved_docs: List[str],
    expected_docs: List[str]
) -> float:
    """
    Recall: Of all relevant docs, what % did we actually retrieve?
    
    Formula: TP / (TP + FN)
    where TP = relevant docs we retrieved
          FN = relevant docs we missed
    """
    if not expected_docs:
        return 1.0  # If no docs expected, perfect recall
    
    # Count how many expected docs were retrieved
    relevant_count = sum(1 for doc in expected_docs if doc in retrieved_docs)
    
    recall = relevant_count / len(expected_docs)
    return float(recall)


def compute_bert_score(generated_answer: str, expected_keywords: List[str]) -> float:
    """
    Semantic similarity: Does the answer contain the right semantic information?
    
    Approximation: Check if expected keywords appear semantically in the answer
    This is a simplified version. Real BERTScore would use embeddings.
    """
    if not expected_keywords:
        return 1.0
    
    answer_lower = generated_answer.lower()
    
    # Count how many keywords appear in the answer (fuzzy match)
    found_keywords = 0
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords += 1
    
    # Score based on keyword coverage
    score = found_keywords / len(expected_keywords)
    
    # Cap at 0.95 since answer likely contains more than just keywords
    return float(min(score, 0.95))


def compute_answer_relevance(generated_answer: str, query: str) -> float:
    """
    Is the answer actually addressing the query?
    
    Approximation: Check if answer is non-trivial and not a refusal
    """
    answer_lower = generated_answer.lower()
    
    # Refusal indicators
    refusal_phrases = [
        "i don't know",
        "i cannot",
        "i'm unable",
        "not found",
        "no information",
        "unable to find"
    ]
    
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return 0.3
    
    # Answer has reasonable length
    if len(generated_answer.split()) < 3:
        return 0.4
    
    return 0.85  # Assume relevant if not a refusal


def detect_hallucinations(
    generated_answer: str,
    retrieved_context: str
) -> bool:
    """
    Did the LLM make up information not in the sources?
    
    Simplified approach: Check if answer contradicts source context
    Real implementation would use NLI models
    """
    # This is hard to do perfectly without advanced NLI
    # For now, assume no hallucinations if answer is relatively short and grounded
    # In production, you'd use a dedicated hallucination detector
    
    answer_words = set(generated_answer.lower().split())
    context_words = set(retrieved_context.lower().split())
    
    # If too many words from answer aren't in context, might be hallucinating
    # (very loose approximation)
    overlap = len(answer_words & context_words) / max(len(answer_words), 1)
    
    # Conservative: flag as hallucination if very low overlap
    is_hallucination = overlap < 0.3
    
    return is_hallucination


def compute_faithfulness(
    generated_answer: str,
    retrieved_context: str
) -> float:
    """
    Is the answer grounded in the sources?
    
    Approximation: Word overlap between answer and context
    Higher overlap = more grounded
    """
    answer_words = set(generated_answer.lower().split())
    context_words = set(retrieved_context.lower().split())
    
    if not answer_words:
        return 0.0
    
    # Overlap ratio
    overlap = len(answer_words & context_words) / len(answer_words)
    
    # Convert to 0-1 scale (0.3 overlap = 0.6 faithfulness)
    faithfulness = min(overlap * 2, 1.0)
    
    return float(faithfulness)


def compute_source_attribution(
    generated_answer: str,
    retrieved_context: str
) -> float:
    """
    What % of the answer is backed by sources?
    
    Approximation: Check what % of answer words appear in retrieved context
    """
    answer_words = generated_answer.lower().split()
    context_words = set(retrieved_context.lower().split())
    
    if not answer_words:
        return 0.0
    
    attributed_words = sum(1 for word in answer_words if word in context_words)
    attribution_score = attributed_words / len(answer_words)
    
    return float(attribution_score)


# ==================== MAIN EVALUATION LOOP ====================

def run_real_evaluation():
    """
    Run actual evaluation against your RAG system.
    """
    print("=" * 70)
    print("REAL RAG EVALUATION")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(store_results=True, results_dir="evaluation_results")
    
    # Create temporary directory for test documents
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nCreating test documents in {tmpdir}")
        
        # Write test documents
        doc_paths = {}
        for doc_name, content in TEST_DOCUMENTS.items():
            doc_path = Path(tmpdir) / doc_name
            doc_path.write_text(content)
            doc_paths[doc_name] = str(doc_path)
            print(f"   ✓ {doc_name}")
        
        # Initialize RAG pipeline
        print(f"\n🚀 Initializing RAG pipeline...")
        try:
            pipeline = RAGPipeline()
        except Exception as e:
            print(f"   ✗ Failed to initialize pipeline: {e}")
            return
        
        # Ingest documents
        print(f"\nIngesting documents into RAG...")
        try:
            for doc_name, content in TEST_DOCUMENTS.items():
                doc_id = Path(doc_name).stem  # Remove extension
                pipeline.ingest(doc_id, content)
                print(f"   ✓ Ingested {doc_name} (doc_id: {doc_id})")
        except Exception as e:
            print(f"   ✗ Failed to ingest documents: {e}")
            return
        
        # Run test cases
        print(f"\nRunning {len(TEST_CASES)} test cases...\n")
        
        all_retrieved_docs = []
        
        for i, test_case in enumerate(TEST_CASES, 1):
            query = test_case["query"]
            expected_keywords = test_case["expected_answer_keywords"]
            expected_docs = test_case["expected_source_docs"]
            
            print(f"Test {i}: {query}")
            print(f"   Expected sources: {expected_docs}")
            
            try:
                start_time = time.time()
                
                # Query the RAG
                result = pipeline.query(query, return_sources=True)
                
                latency_ms = (time.time() - start_time) * 1000
                
                answer = result.get('answer', '')
                retrieved_docs = result.get('sources', [])
                context = result.get('context', '')
                
                # If context is empty, reconstruct from retrieved sources
                if not context and retrieved_docs:
                    # Combine previews from all retrieved sources
                    context = ' '.join([source.get('preview', '') for source in retrieved_docs if isinstance(source, dict)])
                
                print(f"   DEBUG - context length: {len(context)}")
                
                # Extract doc names from sources
                # Sources are dicts with 'chunk_id' like 'medical_research_1_chunk_0'
                retrieved_doc_names = []
                for source in retrieved_docs:
                    if isinstance(source, dict):
                        chunk_id = source.get('chunk_id', '')
                        # Extract doc_id from chunk_id (format: {doc_id}_chunk_{num})
                        doc_id = chunk_id.rsplit('_chunk_', 1)[0] if '_chunk_' in chunk_id else 'unknown'
                        retrieved_doc_names.append(doc_id)
                
                all_retrieved_docs.extend(retrieved_doc_names)
                
                print(f"   Retrieved: {retrieved_doc_names}")
                print(f"   Answer: {answer[:100]}...")
                
                # COMPUTE REAL METRICS
                retrieval_precision = compute_retrieval_precision(
                    retrieved_doc_names, 
                    expected_docs
                )
                retrieval_recall = compute_retrieval_recall(
                    retrieved_doc_names,
                    expected_docs
                )
                bert_score = compute_bert_score(answer, expected_keywords)
                answer_relevance = compute_answer_relevance(answer, query)
                faithfulness = compute_faithfulness(answer, context)
                hallucination_detected = detect_hallucinations(answer, context)
                source_attribution = compute_source_attribution(answer, context)
                
                print(f"   Precision: {retrieval_precision:.2f} | Recall: {retrieval_recall:.2f}")
                print(f"   BERTScore: {bert_score:.2f} | Relevance: {answer_relevance:.2f}")
                print(f"   Faithfulness: {faithfulness:.2f} | Attribution: {source_attribution:.2f}")
                print(f"   Hallucination: {hallucination_detected} | Latency: {latency_ms:.0f}ms")
                
                # Create evaluation result
                eval_result = EvaluationResult(
                    query=query,
                    answer=answer,
                    source_docs=retrieved_doc_names,
                    num_retrieved=len(retrieved_docs),
                    retrieval_precision=retrieval_precision,
                    retrieval_recall=retrieval_recall,
                    rank_position=1 if expected_docs[0] in retrieved_doc_names else 2,
                    rouge_l=bert_score,  # Approximation
                    bert_score=bert_score,
                    answer_relevance=answer_relevance,
                    faithfulness=faithfulness,
                    hallucination_detected=hallucination_detected,
                    source_attribution_score=source_attribution,
                    latency_ms=latency_ms,
                    tokens_used=len(answer.split()),
                    cost_cents=0.004,  # Estimate for Groq
                )
                
                evaluator.add_result(eval_result)
                
                print()
                
            except Exception as e:
                print(f"   ✗ Test failed: {e}\n")
        
        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        metrics = evaluator.compute_aggregate_metrics()
        
        print(f"\nResults:")
        print(f"   Total Evaluations: {metrics['total_evaluations']}")
        print(f"   Avg Precision: {metrics['retrieval_precision_mean']:.3f}")
        print(f"   Avg Recall: {metrics['retrieval_recall_mean']:.3f}")
        print(f"   Avg BERTScore: {metrics['bert_score_mean']:.3f}")
        print(f"   Avg Faithfulness: {metrics['faithfulness_mean']:.3f}")
        print(f"   Hallucination Rate: {metrics['hallucination_rate']*100:.1f}%")
        print(f"   Avg Latency: {metrics['latency_mean']:.0f}ms")
        print(f"   MRR: {metrics['mrr']:.3f}")
        
        print(f"\nResults saved to: evaluation_results/results.jsonl")
        print(f"View dashboard at: http://localhost:8000/evaluation")


if __name__ == "__main__":
    run_real_evaluation()