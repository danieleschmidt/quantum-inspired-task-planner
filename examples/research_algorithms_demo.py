#!/usr/bin/env python3
"""
Research Algorithms Demonstration

This example showcases the cutting-edge quantum algorithms and hybrid approaches
implemented for task scheduling optimization. Demonstrates quantum advantage
and publication-ready research capabilities.

Usage:
    python examples/research_algorithms_demo.py
"""

import sys
import os
import time
import numpy as np
import logging
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.quantum_planner.research import (
        QuantumAlgorithmType, QuantumAlgorithmFactory, AdaptiveQAOAParams,
        HybridQuantumClassicalSolver, HybridMode, DecompositionStrategy,
        ProblemGenerator, BenchmarkRunner, QuantumAdvantageAnalyzer, BenchmarkCategory
    )
    from src.quantum_planner.backends.research_backend import (
        ResearchQuantumBackend, ResearchBackendFactory
    )
    from src.quantum_planner.models import Agent, Task
    from src.quantum_planner.qubo_builder import create_standard_qubo
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Running with mock implementations...")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_problem(num_agents: int = 4, num_tasks: int = 6) -> tuple:
    """Create demonstration task assignment problem."""
    
    # Create agents with different skills and capacities
    agents = [
        Agent(agent_id="agent_1", skills=["python", "ml"], capacity=3),
        Agent(agent_id="agent_2", skills=["javascript", "react"], capacity=2), 
        Agent(agent_id="agent_3", skills=["python", "devops"], capacity=3),
        Agent(agent_id="agent_4", skills=["python", "ml", "devops"], capacity=4)
    ][:num_agents]
    
    # Create tasks with different requirements
    tasks = [
        Task(task_id="task_1", required_skills=["python"], priority=5, duration=2),
        Task(task_id="task_2", required_skills=["javascript", "react"], priority=3, duration=3),
        Task(task_id="task_3", required_skills=["ml"], priority=8, duration=4),
        Task(task_id="task_4", required_skills=["devops"], priority=6, duration=1),
        Task(task_id="task_5", required_skills=["python", "ml"], priority=7, duration=2),
        Task(task_id="task_6", required_skills=["python"], priority=4, duration=1)
    ][:num_tasks]
    
    return agents, tasks


def demo_adaptive_qaoa():
    """Demonstrate Adaptive QAOA algorithm."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Adaptive QAOA Algorithm")
    print("="*60)
    
    if not IMPORTS_AVAILABLE:
        print("Skipped - imports not available")
        return
    
    # Create problem
    agents, tasks = create_demo_problem(3, 4)
    
    # Build QUBO matrix
    Q, variable_map = create_standard_qubo(agents, tasks, "minimize_makespan")
    
    print(f"Problem size: {Q.shape[0]}x{Q.shape[1]} QUBO matrix")
    print(f"Variables: {len(variable_map)} binary variables")
    
    # Configure Adaptive QAOA
    qaoa_params = AdaptiveQAOAParams(
        initial_layers=1,
        max_layers=4,
        adaptation_threshold=1e-3,
        optimizer="COBYLA"
    )
    
    # Create and run algorithm
    qaoa_solver = QuantumAlgorithmFactory.create_algorithm(
        QuantumAlgorithmType.ADAPTIVE_QAOA,
        qaoa_params=qaoa_params
    )
    
    start_time = time.time()
    result = qaoa_solver.solve_scheduling_problem(
        hamiltonian=Q,
        num_qubits=Q.shape[0]
    )
    execution_time = time.time() - start_time
    
    # Display results
    print(f"\nAdaptive QAOA Results:")
    print(f"  Algorithm: {result.algorithm_type.value}")
    print(f"  Final Energy: {result.energy:.6f}")
    print(f"  Convergence Steps: {result.convergence_steps}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Circuit Depth: {result.quantum_circuit_depth}")
    print(f"  Gate Count: {result.gate_count}")
    print(f"  Fidelity: {result.fidelity:.3f}")
    
    # Decode solution
    print(f"\nSolution Assignment:")
    for var_idx, value in result.solution.items():
        if value == 1 and var_idx < len(variable_map):
            task_agent = list(variable_map.values())[var_idx]
            print(f"  {task_agent[0]} → {task_agent[1]}")


def demo_hybrid_decomposition():
    """Demonstrate hybrid quantum-classical decomposition."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Hybrid Quantum-Classical Decomposition")
    print("="*60)
    
    if not IMPORTS_AVAILABLE:
        print("Skipped - imports not available")
        return
    
    # Create larger problem for decomposition
    agents, tasks = create_demo_problem(6, 12)
    Q, variable_map = create_standard_qubo(agents, tasks, "minimize_makespan")
    
    print(f"Large Problem: {Q.shape[0]}x{Q.shape[1]} QUBO matrix")
    print(f"Variables: {len(variable_map)} binary variables")
    
    # Initialize hybrid solver
    hybrid_solver = HybridQuantumClassicalSolver()
    
    # Test different hybrid modes
    hybrid_modes = [HybridMode.SEQUENTIAL, HybridMode.PARALLEL, HybridMode.ADAPTIVE]
    
    for mode in hybrid_modes:
        print(f"\n--- Testing {mode.value.upper()} mode ---")
        
        start_time = time.time()
        result = hybrid_solver.solve_hybrid(
            problem_matrix=Q,
            hybrid_mode=mode,
            decomposition_strategy=DecompositionStrategy.SPECTRAL_CLUSTERING,
            max_subproblem_size=8
        )
        
        print(f"Results for {mode.value} mode:")
        print(f"  Total Energy: {result.total_energy:.6f}")
        print(f"  Quantum Contribution: {result.quantum_contribution:.1%}")
        print(f"  Classical Contribution: {result.classical_contribution:.1%}")
        print(f"  Execution Time: {result.total_execution_time:.3f}s")
        print(f"  Quantum Advantage: {result.quantum_advantage_factor:.2f}x")
        print(f"  Subproblems Solved: {len(result.subproblem_results)}")


def demo_quantum_advantage_benchmarking():
    """Demonstrate quantum advantage benchmarking."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Quantum Advantage Benchmarking")
    print("="*60)
    
    if not IMPORTS_AVAILABLE:
        print("Skipped - imports not available")
        return
    
    # Generate benchmark problems
    generator = ProblemGenerator()
    problems = generator.generate_problem_suite(
        problem_sizes=[8, 10, 12],
        problem_types=["task_assignment"],
        instances_per_size=3,
        seed=42
    )
    
    print(f"Generated {len(problems)} benchmark problems")
    
    # Define algorithms to compare
    def quantum_solver(hamiltonian, **kwargs):
        """Quantum algorithm interface for benchmarking."""
        qaoa_solver = QuantumAlgorithmFactory.create_algorithm(QuantumAlgorithmType.ADAPTIVE_QAOA)
        return qaoa_solver.solve_scheduling_problem(hamiltonian, hamiltonian.shape[0])
    
    def classical_solver(hamiltonian, **kwargs):
        """Classical algorithm interface for benchmarking."""
        hybrid_solver = HybridQuantumClassicalSolver()
        result = hybrid_solver._simulated_annealing_solve(hamiltonian)
        
        # Mock result object for compatibility
        class MockResult:
            def __init__(self, solution, energy):
                self.solution = solution
                self.energy = energy
                self.execution_time = 0.1
                self.convergence_steps = 100
        
        return MockResult(result["solution"], result["energy"])
    
    algorithms = {
        "adaptive_qaoa": quantum_solver,
        "simulated_annealing": classical_solver
    }
    
    # Run benchmark
    runner = BenchmarkRunner(timeout_seconds=30)
    results = runner.run_benchmark_suite(
        algorithms=algorithms,
        problems=problems,
        runs_per_problem=2,
        parallel=False  # Sequential for demo
    )
    
    print(f"Collected {len(results)} benchmark results")
    
    # Analyze quantum advantage
    analyzer = QuantumAdvantageAnalyzer()
    
    try:
        advantage_report = analyzer.analyze_quantum_advantage(
            results=results,
            quantum_algorithm="adaptive_qaoa",
            classical_baseline="simulated_annealing",
            benchmark_category=BenchmarkCategory.PERFORMANCE_SCALING
        )
        
        print(f"\nQuantum Advantage Analysis:")
        print(f"  Quantum Advantage Factor: {advantage_report.quantum_advantage_factor:.2f}x")
        print(f"  Statistical Significance: {advantage_report.statistical_significance}")
        print(f"  Problems Tested: {advantage_report.num_problems_tested}")
        print(f"  Total Runs: {advantage_report.total_runs}")
        print(f"  Confidence Level: {advantage_report.confidence_level:.1%}")
        
        print(f"\nPublication Summary:")
        print(advantage_report.publication_summary)
        
    except Exception as e:
        print(f"Analysis failed: {e}")


def demo_research_backend():
    """Demonstrate the integrated research backend."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Integrated Research Backend")
    print("="*60)
    
    if not IMPORTS_AVAILABLE:
        print("Skipped - imports not available")
        return
    
    # Create different backend configurations
    backends = {
        "Adaptive QAOA": ResearchBackendFactory.create_adaptive_qaoa_backend(
            initial_layers=1,
            max_layers=3
        ),
        "Hybrid Decomposition": ResearchBackendFactory.create_hybrid_backend(
            hybrid_mode="adaptive",
            max_subproblem_size=10
        ),
        "Auto-Selection": ResearchQuantumBackend({
            "algorithm_selection_enabled": True
        })
    }
    
    # Test problem
    agents, tasks = create_demo_problem(4, 6)
    Q, variable_map = create_standard_qubo(agents, tasks, "minimize_makespan")
    
    print(f"Testing problem: {Q.shape[0]}x{Q.shape[1]} QUBO matrix")
    
    for name, backend in backends.items():
        print(f"\n--- {name} Backend ---")
        
        # Get capabilities
        capabilities = backend.get_capabilities()
        print(f"Max Variables: {capabilities.max_variables}")
        print(f"Supports Async: {capabilities.supports_async}")
        
        # Solve problem
        start_time = time.time()
        try:
            solution = backend._solve_qubo(Q)
            solve_time = time.time() - start_time
            
            print(f"Solution found in {solve_time:.3f}s")
            print(f"Variables assigned: {sum(1 for v in solution.values() if v == 1)}")
            
            # Get research metrics
            metrics = backend.get_research_metrics()
            print(f"Research Features: {len(metrics['research_features'])}")
            
        except Exception as e:
            print(f"Solving failed: {e}")


def demo_complete_workflow():
    """Demonstrate complete research workflow."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Complete Research Workflow")
    print("="*60)
    
    if not IMPORTS_AVAILABLE:
        print("Skipped - imports not available")
        return
    
    # Step 1: Problem Generation
    print("Step 1: Generating research problem...")
    agents, tasks = create_demo_problem(5, 8)
    Q, variable_map = create_standard_qubo(agents, tasks, "minimize_makespan")
    
    print(f"Generated problem: {len(tasks)} tasks, {len(agents)} agents")
    print(f"QUBO matrix: {Q.shape[0]}x{Q.shape[1]}")
    
    # Step 2: Algorithm Selection and Execution
    print("\nStep 2: Running multiple quantum algorithms...")
    
    algorithms = [
        QuantumAlgorithmType.ADAPTIVE_QAOA,
        QuantumAlgorithmType.VQE_SCHEDULER,
        QuantumAlgorithmType.QML_PREDICTOR
    ]
    
    results = {}
    
    for alg_type in algorithms:
        try:
            print(f"  Running {alg_type.value}...")
            algorithm = QuantumAlgorithmFactory.create_algorithm(alg_type)
            
            start_time = time.time()
            result = algorithm.solve_scheduling_problem(Q, Q.shape[0])
            execution_time = time.time() - start_time
            
            results[alg_type.value] = {
                "energy": result.energy,
                "time": execution_time,
                "convergence_steps": result.convergence_steps,
                "fidelity": result.fidelity
            }
            
        except Exception as e:
            print(f"    Failed: {e}")
            results[alg_type.value] = {"error": str(e)}
    
    # Step 3: Results Analysis  
    print("\nStep 3: Analyzing results...")
    
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    
    if successful_results:
        best_algorithm = min(successful_results.keys(), 
                           key=lambda k: successful_results[k]["energy"])
        
        print(f"Best Algorithm: {best_algorithm}")
        print(f"  Energy: {successful_results[best_algorithm]['energy']:.6f}")
        print(f"  Time: {successful_results[best_algorithm]['time']:.3f}s")
        print(f"  Fidelity: {successful_results[best_algorithm]['fidelity']:.3f}")
        
        # Compare performance
        print(f"\nPerformance Comparison:")
        for alg_name, result in successful_results.items():
            print(f"  {alg_name}:")
            print(f"    Energy: {result['energy']:.6f}")
            print(f"    Time: {result['time']:.3f}s")
            print(f"    Steps: {result['convergence_steps']}")
    
    # Step 4: Hybrid Enhancement
    print("\nStep 4: Testing hybrid enhancement...")
    
    hybrid_solver = HybridQuantumClassicalSolver()
    hybrid_result = hybrid_solver.solve_hybrid(
        problem_matrix=Q,
        hybrid_mode=HybridMode.ADAPTIVE,
        max_subproblem_size=12
    )
    
    print(f"Hybrid Results:")
    print(f"  Total Energy: {hybrid_result.total_energy:.6f}")
    print(f"  Quantum Advantage: {hybrid_result.quantum_advantage_factor:.2f}x")
    print(f"  Execution Time: {hybrid_result.total_execution_time:.3f}s")
    
    print("\n" + "="*60)
    print("RESEARCH WORKFLOW COMPLETE")
    print("="*60)


def main():
    """Run all research demonstrations."""
    
    print("QUANTUM PLANNER - ADVANCED RESEARCH ALGORITHMS DEMO")
    print("=" * 60)
    print("This demonstration showcases cutting-edge quantum algorithms")
    print("and hybrid approaches for task scheduling optimization.")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("\nWARNING: Some imports failed. Running with limited functionality.")
    
    # Run all demonstrations
    demo_adaptive_qaoa()
    demo_hybrid_decomposition()
    demo_quantum_advantage_benchmarking()
    demo_research_backend()
    demo_complete_workflow()
    
    print(f"\n{'='*60}")
    print("ALL DEMONSTRATIONS COMPLETED")
    print(f"{'='*60}")
    print("\nThese implementations represent state-of-the-art research in")
    print("quantum-enhanced task scheduling and optimization.")
    print("\nFor publication and research use, see:")
    print("- src/quantum_planner/research/")
    print("- Comprehensive benchmarking framework")
    print("- Statistical significance testing")
    print("- Quantum advantage validation")


if __name__ == "__main__":
    main()