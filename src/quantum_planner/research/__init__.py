"""
Quantum Planner Research Module

This module contains cutting-edge quantum algorithms and research implementations
for advanced task scheduling optimization. These implementations are designed
for academic research, publication-quality results, and quantum advantage studies.

Features:
- Novel quantum algorithms (Adaptive QAOA, VQE, Hybrid approaches)
- Comprehensive benchmarking framework
- Statistical significance testing
- Publication-ready result formatting

Usage:
    from quantum_planner.research import (
        NovelQuantumOptimizer, 
        QuantumAlgorithmType,
        novel_optimizer
    )
"""

# Import novel algorithms - main research contribution
try:
    from .novel_quantum_algorithms import (
        NovelQuantumOptimizer,
        QuantumAlgorithmType,
        QuantumCircuitResult,
        ResearchMetrics,
        novel_optimizer
    )
    NOVEL_ALGORITHMS_AVAILABLE = True
except ImportError as e:
    NOVEL_ALGORITHMS_AVAILABLE = False
    print(f"Novel algorithms not available: {e}")
    # Create mock objects to prevent import errors
    class QuantumAlgorithmType:
        pass
    class QuantumCircuitResult:
        pass
    class ResearchMetrics:
        pass
    class NovelQuantumOptimizer:
        pass
    novel_optimizer = None

# Build __all__ list
__all__ = [
    'NovelQuantumOptimizer',
    'QuantumAlgorithmType', 
    'QuantumCircuitResult',
    'ResearchMetrics',
    'novel_optimizer',
    'NOVEL_ALGORITHMS_AVAILABLE'
]