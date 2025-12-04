"""
Performance metrics tracking.
Tracks latency, token usage, and provider statistics.
"""

import time
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    timestamp: datetime
    query: str
    model_used: str
    latency_ms: float
    rag_used: bool
    success: bool
    error: str = None


class MetricsTracker:
    """
    Simple in-memory metrics tracker.
    
    Tracks:
    - Query latency by model
    - RAG vs non-RAG performance
    - Success/error rates
    - Token usage (if available)
    
    For production: Export to Prometheus, DataDog, etc.
    """
    
    def __init__(self):
        self.queries: List[QueryMetrics] = []
        self.model_stats = defaultdict(lambda: {"count": 0, "total_latency": 0, "errors": 0})
        self.rag_stats = {"with_rag": {"count": 0, "total_latency": 0}, "without_rag": {"count": 0, "total_latency": 0}}
    
    def record_query(self, metrics: QueryMetrics):
        """Record metrics for a query."""
        self.queries.append(metrics)
        
        # Update model stats
        model = metrics.model_used
        self.model_stats[model]["count"] += 1
        self.model_stats[model]["total_latency"] += metrics.latency_ms
        if not metrics.success:
            self.model_stats[model]["errors"] += 1
        
        # Update RAG stats
        rag_key = "with_rag" if metrics.rag_used else "without_rag"
        self.rag_stats[rag_key]["count"] += 1
        self.rag_stats[rag_key]["total_latency"] += metrics.latency_ms
    
    def get_summary(self) -> Dict:
        """Get metrics summary."""
        total_queries = len(self.queries)
        successful_queries = sum(1 for q in self.queries if q.success)
        
        # Calculate average latencies
        model_averages = {}
        for model, stats in self.model_stats.items():
            avg_latency = stats["total_latency"] / stats["count"] if stats["count"] > 0 else 0
            model_averages[model] = {
                "count": stats["count"],
                "avg_latency_ms": round(avg_latency, 2),
                "error_rate": round(stats["errors"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0
            }
        
        rag_comparison = {}
        for key, stats in self.rag_stats.items():
            avg_latency = stats["total_latency"] / stats["count"] if stats["count"] > 0 else 0
            rag_comparison[key] = {
                "count": stats["count"],
                "avg_latency_ms": round(avg_latency, 2)
            }
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": round(successful_queries / total_queries * 100, 2) if total_queries > 0 else 0,
            "by_model": model_averages,
            "rag_comparison": rag_comparison
        }


# Global metrics instance
metrics_tracker = MetricsTracker()
