"""
Quantum Planner Research Module

This module contains cutting-edge quantum algorithms and research implementations
for advanced task scheduling optimization. These implementations are designed
for academic research, publication-quality results, and quantum advantage studies.

Features:
- Advanced quantum algorithms (Adaptive QAOA, VQE, QML)
- Comprehensive benchmarking framework
- Hybrid quantum-classical decomposition
- Statistical significance testing
- Publication-ready result formatting

Usage:
    from quantum_planner.research import (
        QuantumAlgorithmFactory, 
        QuantumAdvantageAnalyzer,
        HybridQuantumClassicalSolver
    )
"""

from .advanced_quantum_algorithms import (
    QuantumAlgorithmType,
    QuantumAlgorithmResult,
    AdaptiveQAOAParams,
    AdaptiveQAOAScheduler,
    VQETaskScheduler, 
    QuantumMLTaskPredictor,
    QuantumAlgorithmFactory
)

from .quantum_advantage_benchmarks import (
    BenchmarkCategory,
    AlgorithmClass,
    BenchmarkProblem,
    BenchmarkResult,
    StatisticalAnalysis,
    QuantumAdvantageReport,
    ProblemGenerator,
    BenchmarkRunner,
    QuantumAdvantageAnalyzer
)

from .hybrid_decomposition import (
    DecompositionStrategy,
    HybridMode,
    SubproblemMetrics,
    DecompositionResult,
    HybridSolutionResult,
    ProblemDecomposer,
    HybridQuantumClassicalSolver
)

__all__ = [
    # Advanced Quantum Algorithms
    'QuantumAlgorithmType',
    'QuantumAlgorithmResult', 
    'AdaptiveQAOAParams',
    'AdaptiveQAOAScheduler',
    'VQETaskScheduler',
    'QuantumMLTaskPredictor',
    'QuantumAlgorithmFactory',
    
    # Benchmarking Framework
    'BenchmarkCategory',
    'AlgorithmClass',
    'BenchmarkProblem',
    'BenchmarkResult',
    'StatisticalAnalysis', 
    'QuantumAdvantageReport',
    'ProblemGenerator',
    'BenchmarkRunner',
    'QuantumAdvantageAnalyzer',
    
    # Hybrid Decomposition
    'DecompositionStrategy',
    'HybridMode',
    'SubproblemMetrics',
    'DecompositionResult',
    'HybridSolutionResult',
    'ProblemDecomposer',
    'HybridQuantumClassicalSolver'
]

__version__ = "1.0.0-research"
__author__ = "Terragon Labs Research Team"