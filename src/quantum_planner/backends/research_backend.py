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
from ..research.noise_aware_optimization import (
    AdaptiveNoiseAwareOptimizer, NoiseAwareBackendWrapper, NoiseModelFactory
)
from ..research.realtime_quantum_adaptation import (
    RealTimeAdaptationEngine, PerformanceMetrics, AdaptationTrigger
)
from ..research.quantum_pareto_optimization import (
    MultiObjectiveQuantumOptimizer, MultiObjectiveProblem, MultiObjectiveSolution
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
            "enable_noise_aware_optimization": True,
            "enable_realtime_adaptation": True,
            "enable_multi_objective": True,
            "hybrid_mode": "adaptive",
            "decomposition_strategy": "spectral_clustering",
            "max_subproblem_size": 20,
            "enable_benchmarking": False,
            "qaoa_params": {
                "initial_layers": 2,
                "max_layers": 8,
                "adaptation_threshold": 1e-4
            },
            "noise_profile": "auto_detect",
            "adaptation_rate": 0.1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("research_quantum", default_config)
        
        # Initialize advanced research components
        self.hybrid_solver = HybridQuantumClassicalSolver()
        
        # Initialize noise-aware optimization
        if self.config.get("enable_noise_aware_optimization", True):
            noise_profile = self._initialize_noise_profile()
            self.noise_aware_optimizer = AdaptiveNoiseAwareOptimizer(
                noise_profile=noise_profile,
                adaptation_rate=self.config.get("adaptation_rate", 0.1)
            )
            logger.info("✓ Noise-aware optimization enabled")
        else:
            self.noise_aware_optimizer = None
        
        # Initialize real-time adaptation
        if self.config.get("enable_realtime_adaptation", True):
            self.adaptation_engine = RealTimeAdaptationEngine(
                adaptation_rate=self.config.get("adaptation_rate", 0.1)
            )
            logger.info("✓ Real-time adaptation enabled")
        else:
            self.adaptation_engine = None
        
        # Initialize multi-objective optimizer
        if self.config.get("enable_multi_objective", True):
            self.multi_objective_optimizer = MultiObjectiveQuantumOptimizer(
                quantum_backend=self,
                classical_fallback=True
            )
            logger.info("✓ Multi-objective optimization enabled")
        else:
            self.multi_objective_optimizer = None
        
        # Algorithm selection strategy
        self.algorithm_selection_enabled = True
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Research backend initialized with advanced capabilities")
    
    def _initialize_noise_profile(self):
        """Initialize noise profile based on configuration."""
        noise_profile_config = self.config.get("noise_profile", "auto_detect")
        
        if noise_profile_config == "auto_detect":
            # Auto-detect based on backend type
            return NoiseModelFactory.create_ibm_noise_model("ibmq_qasm_simulator")
        elif isinstance(noise_profile_config, str):
            # Use predefined profile
            if "ibm" in noise_profile_config.lower():
                return NoiseModelFactory.create_ibm_noise_model(noise_profile_config)
            elif "dwave" in noise_profile_config.lower():
                return NoiseModelFactory.create_dwave_noise_model(noise_profile_config)
        
        # Default fallback
        return NoiseModelFactory.create_ibm_noise_model()
    
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
        """Solve QUBO using advanced research algorithms with full integration."""
        
        start_time = time.time()
        
        # Convert to numpy array if needed
        if not isinstance(Q, np.ndarray):
            Q = np.array(Q)
        
        problem_size = Q.shape[0]
        
        self.logger.info(f"Solving {problem_size}x{problem_size} QUBO with research backend")
        
        # Create optimization context for adaptation
        context = self._create_optimization_context(Q, **kwargs)
        
        # Check for multi-objective optimization
        if self._is_multi_objective_problem(kwargs):
            return self._solve_multi_objective(Q, context, **kwargs)
        
        # Apply real-time adaptation if enabled
        if self.adaptation_engine:
            adaptation = self.adaptation_engine.recommend_adaptation(
                context, trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION
            )
            if adaptation:
                self.logger.info(f"Applying adaptation: {adaptation.action_type}")
                # Apply adaptation to context/config
                context = self._apply_adaptation_to_context(context, adaptation)
        
        # Apply noise-aware optimization if enabled
        if self.noise_aware_optimizer:
            # Wrap the solution process with noise-aware optimization
            backend_info = {
                'type': self.backend_name,
                'noise_model': self.noise_aware_optimizer.noise_profile
            }
            
            noise_aware_result = self.noise_aware_optimizer.optimize(Q, backend_info)
            
            # Track performance for adaptation
            self._track_performance(context, noise_aware_result, start_time)
            
            return self._format_solution(noise_aware_result.get('best_solution', {}))
        
        # Determine solution approach based on problem characteristics
        if self.config.get("enable_hybrid_decomposition", True) and problem_size > self.config.get("max_subproblem_size", 20):
            result = self._solve_with_hybrid_decomposition(Q, **kwargs)
        else:
            result = self._solve_with_quantum_algorithm(Q, **kwargs)
        
        # Track performance for adaptation
        end_time = time.time()
        self._track_performance(context, {'best_solution': result}, start_time, end_time)
        
        return result
    
    def _create_optimization_context(self, Q: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Create optimization context for adaptation."""
        return {
            'problem_size': Q.shape[0],
            'noise_level': 0.05,  # Default estimate
            'time_limit': kwargs.get('time_limit', 300),
            'resource_limit': kwargs.get('resource_limit', 1.0),
            'constraints': kwargs.get('constraints', []),
            'current_algorithm': self.config.get('preferred_algorithm', 'adaptive_qaoa'),
            'backend_load': 0.5,  # Default estimate
            'iteration': kwargs.get('iteration', 0),
            'temperature': kwargs.get('temperature', 1.0)
        }
    
    def _is_multi_objective_problem(self, kwargs) -> bool:
        """Check if this is a multi-objective optimization problem."""
        return (
            'objectives' in kwargs and len(kwargs['objectives']) > 1
        ) or (
            'multi_objective' in kwargs and kwargs['multi_objective']
        )
    
    def _solve_multi_objective(self, Q: np.ndarray, context: Dict[str, Any], **kwargs) -> Dict[int, int]:
        """Solve multi-objective problem using quantum Pareto optimization."""
        if not self.multi_objective_optimizer:
            self.logger.warning("Multi-objective optimization not enabled, falling back to single objective")
            return self._solve_with_quantum_algorithm(Q, **kwargs)
        
        # Create multi-objective problem from QUBO
        problem = self._create_multi_objective_problem(Q, **kwargs)
        
        # Optimize
        results = self.multi_objective_optimizer.optimize(
            problem=problem,
            algorithm='quantum_nsga2',
            max_evaluations=kwargs.get('max_evaluations', 5000)
        )
        
        # Extract best solution (e.g., first Pareto front solution)
        pareto_front = results['pareto_front']
        if pareto_front:
            best_solution = pareto_front[0].solution
            return self._format_solution({i: int(best_solution[i]) for i in range(len(best_solution))})
        
        # Fallback to single objective
        return self._solve_with_quantum_algorithm(Q, **kwargs)
    
    def _create_multi_objective_problem(self, Q: np.ndarray, **kwargs) -> MultiObjectiveProblem:
        """Create multi-objective problem from QUBO matrix."""
        
        def objective1(x):
            return x @ Q @ x
        
        def objective2(x):
            # Secondary objective (e.g., minimize number of active variables)
            return np.sum(x)
        
        problem = MultiObjectiveProblem(
            objective_functions=[objective1, objective2],
            objective_names=['energy', 'sparsity'],
            bounds=[(0, 1)] * Q.shape[0],
            variable_types=['binary'] * Q.shape[0],
            num_variables=Q.shape[0]
        )
        
        return problem
    
    def _apply_adaptation_to_context(self, context: Dict[str, Any], adaptation) -> Dict[str, Any]:
        """Apply adaptation action to optimization context."""
        updated_context = context.copy()
        
        if adaptation.action_type == 'algorithm_switch':
            updated_context['current_algorithm'] = adaptation.parameters.get('new_algorithm')
        elif adaptation.action_type == 'parameter_tune':
            param_name = adaptation.parameters.get('param_name')
            new_value = adaptation.parameters.get('new_value')
            updated_context[param_name] = new_value
        elif adaptation.action_type == 'resource_reallocation':
            factor = adaptation.parameters.get('factor', 1.0)
            updated_context['resource_limit'] *= factor
        
        return updated_context
    
    def _track_performance(self, context: Dict[str, Any], result: Dict, start_time: float, end_time: float = None):
        """Track performance for real-time adaptation."""
        if not self.adaptation_engine:
            return
        
        if end_time is None:
            end_time = time.time()
        
        # Extract solution quality metrics
        energy = result.get('best_energy', 0.0)
        solution = result.get('best_solution', {})
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            energy=abs(energy),
            solution_quality=max(0.1, 1.0 / (1.0 + abs(energy))),  # Higher is better
            convergence_rate=0.1,  # Default estimate
            time_to_solution=end_time - start_time,
            resource_utilization=context.get('resource_limit', 1.0),
            noise_resilience=1.0 - context.get('noise_level', 0.05),
            problem_size=context.get('problem_size', 0),
            algorithm_used=context.get('current_algorithm', 'unknown'),
            backend_type='research_quantum',
            noise_level=context.get('noise_level', 0.05)
        )
        
        # Observe performance
        self.adaptation_engine.observe_performance(
            context=context,
            metrics=metrics,
            action_taken=context.get('current_algorithm')
        )
        
        # Store in history
        self.performance_history.append({
            'context': context,
            'metrics': metrics,
            'timestamp': end_time
        })
    
    def _format_solution(self, solution: Dict) -> Dict[int, int]:
        """Format solution to expected output format."""
        if isinstance(solution, dict):
            return {int(k): int(v) for k, v in solution.items()}
        elif isinstance(solution, np.ndarray):
            return {i: int(solution[i]) for i in range(len(solution))}
        else:
            return solution
    
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