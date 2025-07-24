import numpy as np
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor performance metrics for voice conversion"""
    
    def __init__(self):
        self.metrics = {
            "latency_samples": [],
            "cpu_usage": [],
            "memory_usage": [],
            "audio_quality": []
        }
        
    def record_latency(self, latency_ms: float):
        """Record latency measurement"""
        self.metrics["latency_samples"].append(latency_ms)
        
    def get_average_latency(self) -> float:
        """Get average latency"""
        if not self.metrics["latency_samples"]:
            return 0.0
        return np.mean(self.metrics["latency_samples"])
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        latency_samples = self.metrics["latency_samples"]
        return {
            "avg_latency_ms": np.mean(latency_samples) if latency_samples else 0,
            "max_latency_ms": np.max(latency_samples) if latency_samples else 0,
            "min_latency_ms": np.min(latency_samples) if latency_samples else 0,
            "total_samples": len(latency_samples)
        }
        
    def reset_metrics(self):
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key].clear()

