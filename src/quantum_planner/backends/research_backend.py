"""
Research Backend Integration

Integrates advanced research algorithms with the main quantum planner backend system.
Provides seamless access to cutting-edge quantum algorithms and hybrid approaches.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .enhanced_base import EnhancedQuantumBackend, BackendCapabilities, BackendStatus
from ..research.advanced_quantum_algorithms import (
    QuantumAlgorithmType, QuantumAlgorithmFactory, AdaptiveQAOAParams
)
from ..research.hybrid_decomposition import (
    HybridQuantumClassicalSolver, HybridMode, DecompositionStrategy
)

logger = logging.getLogger(__name__)


class ResearchQuantumBackend(EnhancedQuantumBackend):
    """
    Advanced research backend providing access to cutting-edge quantum algorithms.
    
    Features:
    - Adaptive QAOA with dynamic layer optimization
    - VQE with custom scheduling ansätze  
    - Quantum Machine Learning task prediction
    - Hybrid quantum-classical decomposition
    - Automatic algorithm selection based on problem characteristics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize research backend."""
        
        default_config = {
            "preferred_algorithm": "adaptive_qaoa",
            "enable_hybrid_decomposition": True,
            "hybrid_mode": "adaptive",
            "decomposition_strategy": "spectral_clustering",
            "max_subproblem_size": 20,
            "enable_benchmarking": False,
            "qaoa_params": {
                "initial_layers": 2,
                "max_layers": 8,
                "adaptation_threshold": 1e-4
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("research_quantum", default_config)
        
        # Initialize hybrid solver
        self.hybrid_solver = HybridQuantumClassicalSolver()
        
        # Algorithm selection strategy
        self.algorithm_selection_enabled = True
    
    def get_capabilities(self) -> BackendCapabilities:
        """Get research backend capabilities."""
        return BackendCapabilities(
            max_variables=1000,  # Large problems via decomposition
            supports_constraints=True,
            supports_embedding=True,
            supports_async=True,
            supports_batching=True,
            max_batch_size=10,
            supported_objectives=["minimize", "maximize", "multi_objective"],
            constraint_types=["quadratic", "penalty", "custom"]
        )
    
    def _solve_qubo(self, Q: Any, **kwargs) -> Dict[int, int]:
        """Solve QUBO using advanced research algorithms."""
        
        start_time = time.time()
        
        # Convert to numpy array if needed
        if not isinstance(Q, np.ndarray):
            Q = np.array(Q)
        
        problem_size = Q.shape[0]
        
        self.logger.info(f"Solving {problem_size}x{problem_size} QUBO with research backend")
        
        # Determine solution approach based on problem characteristics
        if self.config.get("enable_hybrid_decomposition", True) and problem_size > self.config.get("max_subproblem_size", 20):
            return self._solve_with_hybrid_decomposition(Q, **kwargs)
        else:
            return self._solve_with_quantum_algorithm(Q, **kwargs)
    
    def _solve_with_hybrid_decomposition(self, Q: np.ndarray, **kwargs) -> Dict[int, int]:
        """Solve using hybrid quantum-classical decomposition."""
        
        # Configure hybrid solving
        hybrid_mode_str = self.config.get("hybrid_mode", "adaptive")
        hybrid_mode = HybridMode(hybrid_mode_str)
        
        decomp_strategy_str = self.config.get("decomposition_strategy", "spectral_clustering")  
        decomp_strategy = DecompositionStrategy(decomp_strategy_str)
        
        max_subproblem_size = self.config.get("max_subproblem_size", 20)
        
        self.logger.info(f"Using hybrid decomposition: {hybrid_mode.value} mode, {decomp_strategy.value} strategy")
        
        # Solve with hybrid approach
        hybrid_result = self.hybrid_solver.solve_hybrid(
            problem_matrix=Q,
            hybrid_mode=hybrid_mode,
            decomposition_strategy=decomp_strategy,
            max_subproblem_size=max_subproblem_size,
            **kwargs
        )
        
        self.logger.info(f"Hybrid solution completed: quantum advantage {hybrid_result.quantum_advantage_factor:.2f}x")
        
        return hybrid_result.solution
    
    def _solve_with_quantum_algorithm(self, Q: np.ndarray, **kwargs) -> Dict[int, int]:
        """Solve using single quantum algorithm."""
        
        problem_size = Q.shape[0]
        
        # Algorithm selection
        if self.algorithm_selection_enabled:
            algorithm_type = self._select_optimal_algorithm(Q)
        else:
            preferred = self.config.get("preferred_algorithm", "adaptive_qaoa")
            algorithm_type = QuantumAlgorithmType(preferred)
        
        self.logger.info(f"Selected quantum algorithm: {algorithm_type.value}")
        
        # Create and run algorithm
        if algorithm_type == QuantumAlgorithmType.ADAPTIVE_QAOA:
            qaoa_params = AdaptiveQAOAParams(**self.config.get("qaoa_params", {}))
            algorithm = QuantumAlgorithmFactory.create_algorithm(
                algorithm_type, 
                qaoa_params=qaoa_params
            )
        else:
            algorithm = QuantumAlgorithmFactory.create_algorithm(algorithm_type)
        
        # Execute algorithm
        result = algorithm.solve_scheduling_problem(
            hamiltonian=Q,
            num_qubits=problem_size,
            **kwargs
        )
        
        self.logger.info(f"Quantum algorithm completed: energy = {result.energy:.4f}, steps = {result.convergence_steps}")
        
        return result.solution
    
    def _select_optimal_algorithm(self, Q: np.ndarray) -> QuantumAlgorithmType:
        """Automatically select optimal quantum algorithm based on problem characteristics."""
        
        problem_size = Q.shape[0]
        
        # Calculate problem characteristics
        density = np.count_nonzero(Q) / (Q.shape[0] * Q.shape[1])
        max_coupling = np.max(np.abs(Q))
        avg_coupling = np.mean(np.abs(Q[Q != 0]))
        
        self.logger.debug(f"Problem characteristics: size={problem_size}, density={density:.3f}, max_coupling={max_coupling:.3f}")
        
        # Selection heuristics
        if problem_size <= 12 and density > 0.3:
            # Small, dense problems - QAOA excels
            return QuantumAlgorithmType.ADAPTIVE_QAOA
        
        elif problem_size <= 25 and max_coupling > 5.0:
            # Medium problems with strong couplings - VQE
            return QuantumAlgorithmType.VQE_SCHEDULER
        
        elif problem_size > 25 or density < 0.1:
            # Large or sparse problems - QML may find patterns
            return QuantumAlgorithmType.QML_PREDICTOR
        
        else:
            # Default to adaptive QAOA
            return QuantumAlgorithmType.ADAPTIVE_QAOA
    
    def _check_connectivity(self) -> BackendStatus:
        """Check research backend status."""
        # Research backend is always available (uses simulators/mocks if needed)
        return BackendStatus.AVAILABLE
    
    def _base_time_estimation(self, num_variables: int) -> float:
        """Estimate solving time for research backend."""
        
        if self.config.get("enable_hybrid_decomposition", True) and num_variables > 20:
            # Hybrid decomposition scales better
            return 2.0 + (num_variables / 50) ** 1.5
        else:
            # Pure quantum algorithm scaling
            if num_variables <= 15:
                return 1.0 + num_variables * 0.2
            else:
                return 3.0 + (num_variables / 10) ** 2
    
    def get_research_metrics(self) -> Dict[str, Any]:
        """Get detailed research metrics and statistics."""
        
        return {
            "backend_type": "research_quantum",
            "algorithms_available": [alg.value for alg in QuantumAlgorithmType],
            "hybrid_decomposition_enabled": self.config.get("enable_hybrid_decomposition", True),
            "automatic_algorithm_selection": self.algorithm_selection_enabled,
            "max_problem_size": self.get_capabilities().max_variables,
            "research_features": [
                "adaptive_qaoa",
                "custom_vqe_ansatz", 
                "quantum_ml_prediction",
                "hybrid_decomposition",
                "quantum_advantage_benchmarking"
            ]
        }
    
    def benchmark_quantum_advantage(
        self, 
        problem_sizes: List[int] = [10, 15, 20, 25, 30],
        num_instances: int = 5
    ) -> Dict[str, Any]:
        """Run quantum advantage benchmarking study."""
        
        if not self.config.get("enable_benchmarking", False):
            self.logger.warning("Benchmarking disabled in configuration")
            return {"error": "Benchmarking not enabled"}
        
        try:
            from ..research.quantum_advantage_benchmarks import (
                ProblemGenerator, BenchmarkRunner, QuantumAdvantageAnalyzer, BenchmarkCategory
            )
            
            self.logger.info(f"Starting quantum advantage benchmark: sizes {problem_sizes}, instances {num_instances}")
            
            # Generate benchmark problems
            generator = ProblemGenerator()
            problems = generator.generate_problem_suite(
                problem_sizes=problem_sizes,
                problem_types=["task_assignment"],
                instances_per_size=num_instances
            )
            
            # Define algorithms to compare
            algorithms = {
                "research_quantum": lambda **kwargs: self._solve_qubo(kwargs["hamiltonian"]),
                "classical_sa": self.hybrid_solver._simulated_annealing_solve
            }
            
            # Run benchmarks
            runner = BenchmarkRunner(timeout_seconds=60)
            results = runner.run_benchmark_suite(
                algorithms=algorithms,
                problems=problems,
                runs_per_problem=3
            )
            
            # Analyze results
            analyzer = QuantumAdvantageAnalyzer()
            advantage_report = analyzer.analyze_quantum_advantage(
                results=results,
                quantum_algorithm="research_quantum",
                classical_baseline="classical_sa", 
                benchmark_category=BenchmarkCategory.PERFORMANCE_SCALING
            )
            
            self.logger.info(f"Benchmark completed: {advantage_report.quantum_advantage_factor:.2f}x advantage")
            
            return {
                "quantum_advantage_factor": advantage_report.quantum_advantage_factor,
                "statistical_significance": advantage_report.statistical_significance,
                "num_problems_tested": advantage_report.num_problems_tested,
                "total_runs": advantage_report.total_runs,
                "publication_summary": advantage_report.publication_summary
            }
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            return {"error": str(e)}


class ResearchBackendFactory:
    """Factory for creating research backends with different configurations."""
    
    @staticmethod
    def create_adaptive_qaoa_backend(**qaoa_params) -> ResearchQuantumBackend:
        """Create backend optimized for Adaptive QAOA."""
        
        config = {
            "preferred_algorithm": "adaptive_qaoa",
            "enable_hybrid_decomposition": False,  # Pure QAOA
            "qaoa_params": qaoa_params
        }
        
        return ResearchQuantumBackend(config)
    
    @staticmethod  
    def create_hybrid_backend(
        hybrid_mode: str = "adaptive",
        decomposition_strategy: str = "spectral_clustering",
        max_subproblem_size: int = 20
    ) -> ResearchQuantumBackend:
        """Create backend optimized for hybrid decomposition."""
        
        config = {
            "enable_hybrid_decomposition": True,
            "hybrid_mode": hybrid_mode,
            "decomposition_strategy": decomposition_strategy,
            "max_subproblem_size": max_subproblem_size
        }
        
        return ResearchQuantumBackend(config)
    
    @staticmethod
    def create_benchmarking_backend() -> ResearchQuantumBackend:
        """Create backend with benchmarking enabled."""
        
        config = {
            "enable_benchmarking": True,
            "enable_hybrid_decomposition": True,
            "algorithm_selection_enabled": True
        }
        
        return ResearchQuantumBackend(config)


# Export main interfaces
__all__ = [
    'ResearchQuantumBackend',
    'ResearchBackendFactory'
]