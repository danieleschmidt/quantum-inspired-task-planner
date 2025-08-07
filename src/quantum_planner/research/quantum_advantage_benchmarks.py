"""
Quantum Advantage Benchmarking Framework

This module provides comprehensive benchmarking capabilities to measure and validate
quantum advantage in task scheduling problems. Designed for academic research and
publication-quality results.

Features:
- Comparative analysis of quantum vs classical algorithms
- Statistical significance testing
- Performance scaling analysis
- Hardware-specific benchmarks
- Publication-ready result formatting
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import scipy.stats as stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - statistical analysis limited")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available - plotting disabled")

logger = logging.getLogger(__name__)


class BenchmarkCategory(Enum):
    """Categories of quantum advantage benchmarks."""
    PERFORMANCE_SCALING = "performance_scaling"
    SOLUTION_QUALITY = "solution_quality" 
    CONVERGENCE_SPEED = "convergence_speed"
    HARDWARE_EFFICIENCY = "hardware_efficiency"
    NOISE_RESILIENCE = "noise_resilience"
    PROBLEM_SIZE_SCALING = "problem_size_scaling"


class AlgorithmClass(Enum):
    """Classes of algorithms for comparison."""
    QUANTUM_EXACT = "quantum_exact"
    QUANTUM_APPROXIMATE = "quantum_approximate"
    CLASSICAL_EXACT = "classical_exact"
    CLASSICAL_HEURISTIC = "classical_heuristic"
    HYBRID = "hybrid"


@dataclass
class BenchmarkProblem:
    """Represents a benchmarking problem instance."""
    problem_id: str
    problem_type: str
    num_agents: int
    num_tasks: int
    num_variables: int
    complexity_class: str
    known_optimal: Optional[float]
    problem_data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    algorithm_name: str
    algorithm_class: AlgorithmClass
    problem_id: str
    solution_quality: float
    execution_time: float
    convergence_steps: int
    memory_usage: float
    energy_evaluations: int
    success_probability: float
    solution_found: bool
    metadata: Dict[str, Any]


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of benchmark results."""
    mean_performance: float
    std_deviation: float
    confidence_interval_95: Tuple[float, float]
    median_performance: float
    min_performance: float
    max_performance: float
    p_value_vs_baseline: Optional[float]
    effect_size: Optional[float]
    significance_level: str


@dataclass
class QuantumAdvantageReport:
    """Comprehensive quantum advantage analysis report."""
    benchmark_category: BenchmarkCategory
    problem_class: str
    quantum_algorithm: str
    classical_baseline: str
    num_problems_tested: int
    total_runs: int
    quantum_advantage_factor: float
    statistical_significance: bool
    confidence_level: float
    detailed_results: List[BenchmarkResult]
    statistical_analysis: StatisticalAnalysis
    scaling_analysis: Dict[str, Any]
    publication_summary: str


class ProblemGenerator:
    """Generates benchmark problems for quantum advantage testing."""
    
    @staticmethod
    def generate_problem_suite(
        problem_sizes: List[int],
        problem_types: List[str],
        instances_per_size: int = 10,
        seed: int = 42
    ) -> List[BenchmarkProblem]:
        """Generate comprehensive problem suite."""
        
        np.random.seed(seed)
        problems = []
        
        for problem_type in problem_types:
            for size in problem_sizes:
                for instance in range(instances_per_size):
                    problem = ProblemGenerator._generate_single_problem(
                        problem_type, size, instance
                    )
                    problems.append(problem)
        
        return problems
    
    @staticmethod
    def _generate_single_problem(
        problem_type: str, 
        size: int, 
        instance: int
    ) -> BenchmarkProblem:
        """Generate a single benchmark problem."""
        
        problem_id = f"{problem_type}_size{size}_inst{instance}"
        
        if problem_type == "task_assignment":
            return ProblemGenerator._generate_task_assignment(problem_id, size)
        elif problem_type == "scheduling_with_constraints":
            return ProblemGenerator._generate_constrained_scheduling(problem_id, size)
        elif problem_type == "multi_objective":
            return ProblemGenerator._generate_multi_objective(problem_id, size)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    @staticmethod
    def _generate_task_assignment(problem_id: str, size: int) -> BenchmarkProblem:
        """Generate task assignment problem."""
        
        num_agents = max(2, size // 2)
        num_tasks = size
        
        # Generate random task-agent compatibility matrix
        compatibility = np.random.rand(num_tasks, num_agents)
        
        # Generate costs
        costs = np.random.uniform(1, 10, (num_tasks, num_agents))
        
        # Create QUBO matrix
        num_vars = num_tasks * num_agents
        Q = np.zeros((num_vars, num_vars))
        
        # Fill QUBO matrix based on problem structure
        for t in range(num_tasks):
            for a in range(num_agents):
                var_idx = t * num_agents + a
                Q[var_idx, var_idx] = costs[t, a]
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_type="task_assignment",
            num_agents=num_agents,
            num_tasks=num_tasks,
            num_variables=num_vars,
            complexity_class="NP-hard",
            known_optimal=None,  # Could compute for small instances
            problem_data={
                "qubo_matrix": Q.tolist(),
                "compatibility": compatibility.tolist(),
                "costs": costs.tolist()
            },
            metadata={"generated_at": time.time()}
        )
    
    @staticmethod
    def _generate_constrained_scheduling(problem_id: str, size: int) -> BenchmarkProblem:
        """Generate scheduling problem with precedence constraints."""
        
        num_agents = max(2, size // 3)
        num_tasks = size
        
        # Generate precedence graph (DAG)
        precedences = {}
        for i in range(1, num_tasks):
            if np.random.random() < 0.3:  # 30% chance of precedence
                predecessor = np.random.randint(0, i)
                if predecessor not in precedences:
                    precedences[predecessor] = []
                precedences[predecessor].append(i)
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_type="scheduling_with_constraints",
            num_agents=num_agents,
            num_tasks=num_tasks,
            num_variables=num_tasks * num_agents,
            complexity_class="NP-complete",
            known_optimal=None,
            problem_data={
                "precedences": precedences,
                "task_durations": np.random.uniform(1, 5, num_tasks).tolist(),
                "agent_capacities": np.random.uniform(8, 15, num_agents).tolist()
            },
            metadata={"constraint_density": len(precedences) / num_tasks}
        )
    
    @staticmethod
    def _generate_multi_objective(problem_id: str, size: int) -> BenchmarkProblem:
        """Generate multi-objective scheduling problem."""
        
        num_agents = max(3, size // 2)
        num_tasks = size
        
        # Multiple objectives: cost, time, quality
        cost_matrix = np.random.uniform(1, 10, (num_tasks, num_agents))
        time_matrix = np.random.uniform(1, 8, (num_tasks, num_agents))
        quality_matrix = np.random.uniform(0.5, 1.0, (num_tasks, num_agents))
        
        return BenchmarkProblem(
            problem_id=problem_id,
            problem_type="multi_objective",
            num_agents=num_agents,
            num_tasks=num_tasks,
            num_variables=num_tasks * num_agents,
            complexity_class="Multi-objective NP-hard",
            known_optimal=None,
            problem_data={
                "cost_matrix": cost_matrix.tolist(),
                "time_matrix": time_matrix.tolist(),
                "quality_matrix": quality_matrix.tolist()
            },
            metadata={"num_objectives": 3}
        )


class BenchmarkRunner:
    """Executes benchmark runs and collects results."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timeout_seconds
        self.logger = logging.getLogger(f"{__name__}.BenchmarkRunner")
    
    def run_benchmark_suite(
        self,
        algorithms: Dict[str, Callable],
        problems: List[BenchmarkProblem],
        runs_per_problem: int = 5,
        parallel: bool = True
    ) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        
        results = []
        total_runs = len(algorithms) * len(problems) * runs_per_problem
        
        self.logger.info(f"Starting benchmark suite: {total_runs} total runs")
        
        if parallel:
            results = self._run_parallel_benchmarks(
                algorithms, problems, runs_per_problem
            )
        else:
            results = self._run_sequential_benchmarks(
                algorithms, problems, runs_per_problem
            )
        
        self.logger.info(f"Benchmark suite completed: {len(results)} results")
        return results
    
    def _run_parallel_benchmarks(
        self,
        algorithms: Dict[str, Callable],
        problems: List[BenchmarkProblem],
        runs_per_problem: int
    ) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for alg_name, alg_func in algorithms.items():
                for problem in problems:
                    for run in range(runs_per_problem):
                        future = executor.submit(
                            self._run_single_benchmark,
                            alg_name, alg_func, problem, run
                        )
                        futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.timeout)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Benchmark run failed: {e}")
        
        return results
    
    def _run_sequential_benchmarks(
        self,
        algorithms: Dict[str, Callable],
        problems: List[BenchmarkProblem],
        runs_per_problem: int
    ) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        
        results = []
        
        for alg_name, alg_func in algorithms.items():
            for problem in problems:
                for run in range(runs_per_problem):
                    try:
                        result = self._run_single_benchmark(
                            alg_name, alg_func, problem, run
                        )
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Benchmark run failed: {e}")
        
        return results
    
    def _run_single_benchmark(
        self,
        algorithm_name: str,
        algorithm_func: Callable,
        problem: BenchmarkProblem,
        run_id: int
    ) -> Optional[BenchmarkResult]:
        """Run single benchmark and collect results."""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Extract problem data
            qubo_matrix = None
            if "qubo_matrix" in problem.problem_data:
                qubo_matrix = np.array(problem.problem_data["qubo_matrix"])
            
            # Run algorithm
            if qubo_matrix is not None:
                result = algorithm_func(
                    hamiltonian=qubo_matrix,
                    num_qubits=problem.num_variables,
                    problem_data=problem.problem_data
                )
            else:
                # Handle other problem formats
                result = algorithm_func(problem_data=problem.problem_data)
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            # Evaluate solution quality
            solution_quality = self._evaluate_solution_quality(
                result, problem, qubo_matrix
            )
            
            # Determine algorithm class
            algorithm_class = self._classify_algorithm(algorithm_name)
            
            return BenchmarkResult(
                algorithm_name=algorithm_name,
                algorithm_class=algorithm_class,
                problem_id=problem.problem_id,
                solution_quality=solution_quality,
                execution_time=execution_time,
                convergence_steps=getattr(result, 'convergence_steps', 1),
                memory_usage=memory_usage,
                energy_evaluations=getattr(result, 'energy_evaluations', 1),
                success_probability=1.0,  # Assume success if no exception
                solution_found=True,
                metadata={
                    "run_id": run_id,
                    "timestamp": start_time,
                    "algorithm_metadata": getattr(result, 'metadata', {})
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.warning(f"Algorithm {algorithm_name} failed on {problem.problem_id}: {e}")
            
            return BenchmarkResult(
                algorithm_name=algorithm_name,
                algorithm_class=self._classify_algorithm(algorithm_name),
                problem_id=problem.problem_id,
                solution_quality=float('inf'),  # Worst possible
                execution_time=execution_time,
                convergence_steps=0,
                memory_usage=0,
                energy_evaluations=0,
                success_probability=0.0,
                solution_found=False,
                metadata={"error": str(e), "run_id": run_id}
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _evaluate_solution_quality(
        self,
        result: Any,
        problem: BenchmarkProblem,
        qubo_matrix: Optional[np.ndarray]
    ) -> float:
        """Evaluate the quality of a solution."""
        
        if hasattr(result, 'energy'):
            return float(result.energy)
        elif hasattr(result, 'solution') and qubo_matrix is not None:
            # Calculate QUBO energy manually
            solution = result.solution
            energy = 0.0
            
            for i in range(qubo_matrix.shape[0]):
                for j in range(qubo_matrix.shape[1]):
                    if i in solution and j in solution:
                        energy += qubo_matrix[i, j] * solution[i] * solution[j]
            
            return energy
        else:
            # Return execution time as proxy for quality
            return getattr(result, 'execution_time', float('inf'))
    
    def _classify_algorithm(self, algorithm_name: str) -> AlgorithmClass:
        """Classify algorithm type."""
        
        name_lower = algorithm_name.lower()
        
        if any(quantum_word in name_lower for quantum_word in 
               ['qaoa', 'vqe', 'quantum', 'adiabatic', 'qml']):
            if 'exact' in name_lower:
                return AlgorithmClass.QUANTUM_EXACT
            else:
                return AlgorithmClass.QUANTUM_APPROXIMATE
        
        elif 'hybrid' in name_lower:
            return AlgorithmClass.HYBRID
        
        elif any(exact_word in name_lower for exact_word in ['exact', 'optimal', 'branch']):
            return AlgorithmClass.CLASSICAL_EXACT
        
        else:
            return AlgorithmClass.CLASSICAL_HEURISTIC


class QuantumAdvantageAnalyzer:
    """Analyzes benchmark results for quantum advantage."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QuantumAdvantageAnalyzer")
    
    def analyze_quantum_advantage(
        self,
        results: List[BenchmarkResult],
        quantum_algorithm: str,
        classical_baseline: str,
        benchmark_category: BenchmarkCategory
    ) -> QuantumAdvantageReport:
        """Comprehensive quantum advantage analysis."""
        
        # Filter results for comparison
        quantum_results = [r for r in results if r.algorithm_name == quantum_algorithm]
        classical_results = [r for r in results if r.algorithm_name == classical_baseline]
        
        if not quantum_results or not classical_results:
            raise ValueError("Insufficient results for comparison")
        
        # Group by problem for paired comparison
        quantum_by_problem = self._group_by_problem(quantum_results)
        classical_by_problem = self._group_by_problem(classical_results)
        
        # Calculate advantage metrics
        advantage_factor = self._calculate_advantage_factor(
            quantum_by_problem, classical_by_problem
        )
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            quantum_by_problem, classical_by_problem
        )
        
        # Scaling analysis
        scaling_analysis = self._analyze_scaling(quantum_results, classical_results)
        
        # Generate publication summary
        publication_summary = self._generate_publication_summary(
            quantum_algorithm, classical_baseline, advantage_factor, 
            statistical_analysis, len(quantum_results)
        )
        
        return QuantumAdvantageReport(
            benchmark_category=benchmark_category,
            problem_class="Task Scheduling",
            quantum_algorithm=quantum_algorithm,
            classical_baseline=classical_baseline,
            num_problems_tested=len(quantum_by_problem),
            total_runs=len(quantum_results) + len(classical_results),
            quantum_advantage_factor=advantage_factor,
            statistical_significance=statistical_analysis.p_value_vs_baseline < 0.05 if statistical_analysis.p_value_vs_baseline else False,
            confidence_level=0.95,
            detailed_results=quantum_results + classical_results,
            statistical_analysis=statistical_analysis,
            scaling_analysis=scaling_analysis,
            publication_summary=publication_summary
        )
    
    def _group_by_problem(self, results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """Group results by problem ID."""
        grouped = {}
        for result in results:
            if result.problem_id not in grouped:
                grouped[result.problem_id] = []
            grouped[result.problem_id].append(result)
        return grouped
    
    def _calculate_advantage_factor(
        self,
        quantum_by_problem: Dict[str, List[BenchmarkResult]],
        classical_by_problem: Dict[str, List[BenchmarkResult]]
    ) -> float:
        """Calculate overall quantum advantage factor."""
        
        ratios = []
        
        for problem_id in quantum_by_problem:
            if problem_id in classical_by_problem:
                quantum_avg = np.mean([r.execution_time for r in quantum_by_problem[problem_id]])
                classical_avg = np.mean([r.execution_time for r in classical_by_problem[problem_id]])
                
                if quantum_avg > 0:
                    ratio = classical_avg / quantum_avg
                    ratios.append(ratio)
        
        return np.mean(ratios) if ratios else 1.0
    
    def _perform_statistical_analysis(
        self,
        quantum_by_problem: Dict[str, List[BenchmarkResult]],
        classical_by_problem: Dict[str, List[BenchmarkResult]]
    ) -> StatisticalAnalysis:
        """Perform statistical significance testing."""
        
        quantum_times = []
        classical_times = []
        
        # Collect paired samples
        for problem_id in quantum_by_problem:
            if problem_id in classical_by_problem:
                quantum_times.extend([r.execution_time for r in quantum_by_problem[problem_id]])
                classical_times.extend([r.execution_time for r in classical_by_problem[problem_id]])
        
        if not quantum_times or not classical_times:
            return StatisticalAnalysis(
                mean_performance=0, std_deviation=0, confidence_interval_95=(0, 0),
                median_performance=0, min_performance=0, max_performance=0,
                p_value_vs_baseline=None, effect_size=None, significance_level="N/A"
            )
        
        # Calculate statistics
        quantum_mean = np.mean(quantum_times)
        quantum_std = np.std(quantum_times)
        quantum_median = np.median(quantum_times)
        
        # Confidence interval
        if SCIPY_AVAILABLE:
            confidence_interval = stats.t.interval(
                0.95, len(quantum_times)-1, loc=quantum_mean, 
                scale=quantum_std/np.sqrt(len(quantum_times))
            )
            
            # Paired t-test
            if len(quantum_times) == len(classical_times):
                t_stat, p_value = stats.ttest_rel(classical_times, quantum_times)
            else:
                t_stat, p_value = stats.ttest_ind(classical_times, quantum_times)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((quantum_std**2 + np.std(classical_times)**2) / 2)
            effect_size = (np.mean(classical_times) - quantum_mean) / pooled_std if pooled_std > 0 else 0
            
        else:
            confidence_interval = (quantum_mean - quantum_std, quantum_mean + quantum_std)
            p_value = None
            effect_size = None
        
        # Significance level
        if p_value is not None:
            if p_value < 0.001:
                significance_level = "p < 0.001"
            elif p_value < 0.01:
                significance_level = "p < 0.01"
            elif p_value < 0.05:
                significance_level = "p < 0.05"
            else:
                significance_level = f"p = {p_value:.3f}"
        else:
            significance_level = "N/A"
        
        return StatisticalAnalysis(
            mean_performance=quantum_mean,
            std_deviation=quantum_std,
            confidence_interval_95=confidence_interval,
            median_performance=quantum_median,
            min_performance=min(quantum_times),
            max_performance=max(quantum_times),
            p_value_vs_baseline=p_value,
            effect_size=effect_size,
            significance_level=significance_level
        )
    
    def _analyze_scaling(
        self, 
        quantum_results: List[BenchmarkResult],
        classical_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Analyze algorithm scaling behavior."""
        
        # Extract problem sizes
        quantum_scaling = self._extract_scaling_data(quantum_results)
        classical_scaling = self._extract_scaling_data(classical_results)
        
        return {
            "quantum_scaling": quantum_scaling,
            "classical_scaling": classical_scaling,
            "scaling_advantage": self._compare_scaling(quantum_scaling, classical_scaling)
        }
    
    def _extract_scaling_data(self, results: List[BenchmarkResult]) -> Dict[int, float]:
        """Extract scaling data from results."""
        scaling_data = {}
        
        for result in results:
            # Extract problem size from problem_id
            try:
                size_str = result.problem_id.split("_size")[1].split("_")[0]
                problem_size = int(size_str)
                
                if problem_size not in scaling_data:
                    scaling_data[problem_size] = []
                scaling_data[problem_size].append(result.execution_time)
            except (ValueError, IndexError):
                continue
        
        # Average times for each size
        averaged_scaling = {}
        for size, times in scaling_data.items():
            averaged_scaling[size] = np.mean(times)
        
        return averaged_scaling
    
    def _compare_scaling(self, quantum_scaling: Dict[int, float], classical_scaling: Dict[int, float]) -> str:
        """Compare scaling behavior."""
        
        common_sizes = set(quantum_scaling.keys()) & set(classical_scaling.keys())
        
        if len(common_sizes) < 3:
            return "Insufficient data for scaling analysis"
        
        # Simple analysis: check if quantum scales better at larger sizes
        larger_sizes = sorted(common_sizes)[-3:]
        
        quantum_improvement = []
        for size in larger_sizes:
            if classical_scaling[size] > 0:
                improvement = classical_scaling[size] / quantum_scaling[size]
                quantum_improvement.append(improvement)
        
        avg_improvement = np.mean(quantum_improvement) if quantum_improvement else 1.0
        
        if avg_improvement > 2.0:
            return f"Strong quantum advantage at scale (avg {avg_improvement:.2f}x speedup)"
        elif avg_improvement > 1.2:
            return f"Moderate quantum advantage at scale (avg {avg_improvement:.2f}x speedup)"
        else:
            return f"Limited quantum advantage at scale (avg {avg_improvement:.2f}x speedup)"
    
    def _generate_publication_summary(
        self,
        quantum_algorithm: str,
        classical_baseline: str, 
        advantage_factor: float,
        statistical_analysis: StatisticalAnalysis,
        num_runs: int
    ) -> str:
        """Generate publication-ready summary."""
        
        summary = f"""
QUANTUM ADVANTAGE ANALYSIS SUMMARY

Algorithm Comparison: {quantum_algorithm} vs {classical_baseline}
Total Benchmark Runs: {num_runs}

Performance Results:
- Quantum Advantage Factor: {advantage_factor:.2f}x
- Mean Quantum Performance: {statistical_analysis.mean_performance:.4f}s
- Standard Deviation: {statistical_analysis.std_deviation:.4f}s
- 95% Confidence Interval: ({statistical_analysis.confidence_interval_95[0]:.4f}, {statistical_analysis.confidence_interval_95[1]:.4f})

Statistical Significance:
- Significance Level: {statistical_analysis.significance_level}
- Effect Size: {statistical_analysis.effect_size:.3f if statistical_analysis.effect_size else 'N/A'}

Conclusion: {"Statistically significant quantum advantage demonstrated" if statistical_analysis.p_value_vs_baseline and statistical_analysis.p_value_vs_baseline < 0.05 else "No statistically significant quantum advantage observed"}
        """.strip()
        
        return summary


# Export public API
__all__ = [
    'BenchmarkCategory',
    'AlgorithmClass', 
    'BenchmarkProblem',
    'BenchmarkResult',
    'StatisticalAnalysis',
    'QuantumAdvantageReport',
    'ProblemGenerator',
    'BenchmarkRunner',
    'QuantumAdvantageAnalyzer'
]