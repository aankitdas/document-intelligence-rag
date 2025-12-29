"""
RAG Evaluation Module
Comprehensive evaluation metrics for Retrieval-Augmented Generation systems.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path


@dataclass
class EvaluationResult:
    """Single evaluation result for a query-answer pair."""
    query: str
    answer: str
    source_docs: List[str]
    
    # Retrieval metrics
    num_retrieved: int
    retrieval_precision: float
    retrieval_recall: float
    rank_position: int  # Position of correct doc in ranked results
    
    # Generation metrics
    rouge_l: float  # Token-level overlap
    bert_score: float  # Semantic similarity
    answer_relevance: float  # Is answer relevant to query?
    
    # Faithfulness metrics
    faithfulness: float  # Is answer grounded in sources?
    hallucination_detected: bool
    source_attribution_score: float  # % of answer backed by sources
    
    # Performance metrics
    latency_ms: float
    tokens_used: int
    cost_cents: float
    
    # Metadata
    timestamp: str = ""
    eval_id: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.eval_id:
            # Generate unique ID from query hash
            self.eval_id = hashlib.md5(f"{self.query}{self.timestamp}".encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # data['hallucination_detected'] = int(data['hallucination_detected'])
        return data


class RAGEvaluator:
    """Main evaluation engine for RAG systems."""
    
    def __init__(self, store_results: bool = True, results_dir: str = "evaluation_results"):
        """
        Initialize evaluator.
        
        Args:
            store_results: Whether to store results to disk
            results_dir: Directory to store evaluation results
        """
        BASE_DIR = Path(__file__).resolve().parents[2]
        self.store_results = store_results
        self.results_dir = BASE_DIR / results_dir
        self.results_dir.mkdir(exist_ok=True)
        self.results: List[EvaluationResult] = []
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load existing results from disk."""
        results_file = self.results_dir / "results.jsonl"
        print("CWD:", Path.cwd())
        print("Expected results path:", (self.results_dir / "results.jsonl").resolve())
        print("Exists:", (self.results_dir / "results.jsonl").exists())
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        data['hallucination_detected'] = bool(data['hallucination_detected'])
                        self.results.append(EvaluationResult(**data))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load evaluation results from {results_file}"
                ) from e
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add evaluation result."""
        self.results.append(result)
        if self.store_results:
            self._save_result(result)
    
    def _save_result(self, result: EvaluationResult) -> None:
        """Save single result to disk."""
        results_file = self.results_dir / "results.jsonl"
        try:
            with open(results_file, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
        except Exception as e:
            print(f"Warning: Could not save result: {e}")
    
    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all results."""
        if not self.results:
            return self._empty_metrics()
        
        results_data = [r.to_dict() for r in self.results]
        
        # Convert to numeric arrays
        retrieval_precision = np.array([r['retrieval_precision'] for r in results_data])
        retrieval_recall = np.array([r['retrieval_recall'] for r in results_data])
        rouge_l = np.array([r['rouge_l'] for r in results_data])
        bert_score = np.array([r['bert_score'] for r in results_data])
        faithfulness = np.array([r['faithfulness'] for r in results_data])
        answer_relevance = np.array([r['answer_relevance'] for r in results_data])
        latency = np.array([r['latency_ms'] for r in results_data])
        costs = np.array([r['cost_cents'] for r in results_data])
        rank_pos = np.array([r['rank_position'] for r in results_data])
        hallucinations = np.array([r['hallucination_detected'] for r in results_data])
        source_attr = np.array([r['source_attribution_score'] for r in results_data])
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = np.mean(1.0 / rank_pos)
        
        return {
            # Retrieval Metrics
            "retrieval_precision_mean": float(np.mean(retrieval_precision)),
            "retrieval_precision_std": float(np.std(retrieval_precision)),
            "retrieval_recall_mean": float(np.mean(retrieval_recall)),
            "retrieval_recall_std": float(np.std(retrieval_recall)),
            "mrr": float(mrr),
            
            # Generation Metrics
            "rouge_l_mean": float(np.mean(rouge_l)),
            "rouge_l_std": float(np.std(rouge_l)),
            "bert_score_mean": float(np.mean(bert_score)),
            "bert_score_std": float(np.std(bert_score)),
            "answer_relevance_mean": float(np.mean(answer_relevance)),
            "answer_relevance_std": float(np.std(answer_relevance)),
            
            # Faithfulness Metrics
            "faithfulness_mean": float(np.mean(faithfulness)),
            "faithfulness_std": float(np.std(faithfulness)),
            "hallucination_rate": float(np.sum(hallucinations) / len(hallucinations)),
            "source_attribution_mean": float(np.mean(source_attr)),
            "source_attribution_std": float(np.std(source_attr)),
            
            # Performance Metrics
            "latency_p50": float(np.percentile(latency, 50)),
            "latency_p95": float(np.percentile(latency, 95)),
            "latency_p99": float(np.percentile(latency, 99)),
            "latency_mean": float(np.mean(latency)),
            "latency_std": float(np.std(latency)),
            "cost_per_query": float(np.mean(costs)),
            "total_cost": float(np.sum(costs)),
            
            # Metadata
            "total_evaluations": len(self.results),
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_results_timeseries(self) -> Dict[str, List[Any]]:
        """Get results as timeseries for visualization."""
        results_data = [r.to_dict() for r in self.results]
        
        if not results_data:
            return {}
        
        timeseries = {
            "query_idx": list(range(len(results_data))),
            "retrieval_precision": [r['retrieval_precision'] for r in results_data],
            "retrieval_recall": [r['retrieval_recall'] for r in results_data],
            "rouge_l": [r['rouge_l'] for r in results_data],
            "bert_score": [r['bert_score'] for r in results_data],
            "faithfulness": [r['faithfulness'] for r in results_data],
            "answer_relevance": [r['answer_relevance'] for r in results_data],
            "latency_ms": [r['latency_ms'] for r in results_data],
            "hallucination": [int(r['hallucination_detected']) for r in results_data],
        }
        
        return timeseries
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failure modes."""
        if not self.results:
            return self._empty_failure_analysis()
        
        results_data = [r.to_dict() for r in self.results]
        
        # Define failure thresholds
        low_retrieval_threshold = np.median([r['retrieval_precision'] for r in results_data]) * 0.7
        low_generation_threshold = np.median([r['bert_score'] for r in results_data]) * 0.7
        low_faithfulness_threshold = 0.8
        
        failures = {
            "hallucinations": [],
            "low_retrieval": [],
            "low_generation": [],
            "low_faithfulness": [],
        }
        
        for r in results_data:
            if r['hallucination_detected']:
                failures["hallucinations"].append({
                    "eval_id": r['eval_id'],
                    "query": r['query'][:100],
                    "score": r['faithfulness']
                })
            
            if r['retrieval_precision'] < low_retrieval_threshold:
                failures["low_retrieval"].append({
                    "eval_id": r['eval_id'],
                    "query": r['query'][:100],
                    "score": r['retrieval_precision']
                })
            
            if r['bert_score'] < low_generation_threshold:
                failures["low_generation"].append({
                    "eval_id": r['eval_id'],
                    "query": r['query'][:100],
                    "score": r['bert_score']
                })
            
            if r['faithfulness'] < low_faithfulness_threshold:
                failures["low_faithfulness"].append({
                    "eval_id": r['eval_id'],
                    "query": r['query'][:100],
                    "score": r['faithfulness']
                })
        
        return {
            "total_failures": sum(len(v) for v in failures.values()),
            "failure_modes": {k: len(v) for k, v in failures.items()},
            "failure_details": failures,
        }
    
    def get_percentile_analysis(self) -> Dict[str, Any]:
        """Get percentile analysis for performance metrics."""
        if not self.results:
            return {}
        
        results_data = [r.to_dict() for r in self.results]
        
        metrics_to_analyze = {
            "retrieval_precision": [r['retrieval_precision'] for r in results_data],
            "retrieval_recall": [r['retrieval_recall'] for r in results_data],
            "rouge_l": [r['rouge_l'] for r in results_data],
            "bert_score": [r['bert_score'] for r in results_data],
            "faithfulness": [r['faithfulness'] for r in results_data],
            "latency_ms": [r['latency_ms'] for r in results_data],
        }
        
        percentile_analysis = {}
        for metric_name, values in metrics_to_analyze.items():
            percentile_analysis[metric_name] = {
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
        
        return percentile_analysis
    
    def export_to_csv(self, filepath: str) -> None:
        """Export results to CSV."""
        if not self.results:
            print("No results to export")
            return
        
        import csv
        
        results_data = [r.to_dict() for r in self.results]
        
        if results_data:
            keys = results_data[0].keys()
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results_data)
            print(f"Exported {len(results_data)} results to {filepath}")
    
    def reset(self) -> None:
        """Clear all results."""
        self.results = []
        results_file = self.results_dir / "results.jsonl"
        if results_file.exists():
            results_file.unlink()
    
    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "retrieval_precision_mean": 0.0,
            "retrieval_precision_std": 0.0,
            "retrieval_recall_mean": 0.0,
            "retrieval_recall_std": 0.0,
            "mrr": 0.0,
            "rouge_l_mean": 0.0,
            "rouge_l_std": 0.0,
            "bert_score_mean": 0.0,
            "bert_score_std": 0.0,
            "answer_relevance_mean": 0.0,
            "answer_relevance_std": 0.0,
            "faithfulness_mean": 0.0,
            "faithfulness_std": 0.0,
            "hallucination_rate": 0.0,
            "source_attribution_mean": 0.0,
            "source_attribution_std": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
            "latency_mean": 0.0,
            "latency_std": 0.0,
            "cost_per_query": 0.0,
            "total_cost": 0.0,
            "total_evaluations": 0,
            "timestamp": datetime.now().isoformat(),
        }
    
    @staticmethod
    def _empty_failure_analysis() -> Dict[str, Any]:
        """Return empty failure analysis."""
        return {
            "total_failures": 0,
            "failure_modes": {
                "hallucinations": 0,
                "low_retrieval": 0,
                "low_generation": 0,
                "low_faithfulness": 0,
            },
            "failure_details": {
                "hallucinations": [],
                "low_retrieval": [],
                "low_generation": [],
                "low_faithfulness": [],
            },
        }