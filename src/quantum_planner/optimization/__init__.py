"""Optimization and performance utilities for quantum task planning."""

from .performance import (
    PerformanceConfig,
    ProblemStats,
    IntelligentCache,
    ProblemDecomposer,
    LoadBalancer,
    ParallelSolver
)

try:
    from .enhanced_performance import (
        AdaptiveWorkloadBalancer,
        PredictiveResourceAllocator, 
        RealTimePerformanceTuner,
        QuantumClassicalHybridOptimizer,
        PerformanceMetrics
    )
    from .research_integration import (
        NeuralQuantumFusionOptimizer,
        StatisticalValidationEngine,
        ResearchBenchmark,
        ExperimentalResults
    )
    _has_enhanced = True
except ImportError:
    _has_enhanced = False

__all__ = [
    "PerformanceConfig",
    "ProblemStats", 
    "IntelligentCache",
    "ProblemDecomposer",
    "LoadBalancer",
    "ParallelSolver"
]

if _has_enhanced:
    __all__.extend([
        "AdaptiveWorkloadBalancer",
        "PredictiveResourceAllocator",
        "RealTimePerformanceTuner", 
        "QuantumClassicalHybridOptimizer",
        "PerformanceMetrics",
        "NeuralQuantumFusionOptimizer",
        "StatisticalValidationEngine",
        "ResearchBenchmark",
        "ExperimentalResults"
    ])