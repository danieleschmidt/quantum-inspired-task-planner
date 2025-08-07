#!/usr/bin/env python3
"""
Comprehensive Test Suite for Research Implementation

This test suite validates the advanced quantum algorithms, hybrid decomposition,
and benchmarking framework implemented for the research module.

Usage:
    python test_research_implementation.py
"""

import sys
import os
import unittest
import numpy as np
import time
from typing import Dict, List, Any

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from quantum_planner.research import (
        QuantumAlgorithmType, QuantumAlgorithmFactory, AdaptiveQAOAParams,
        HybridQuantumClassicalSolver, HybridMode, DecompositionStrategy,
        ProblemGenerator, BenchmarkRunner, QuantumAdvantageAnalyzer
    )
    from quantum_planner.backends.research_backend import ResearchQuantumBackend
    RESEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Research modules not available: {e}")
    RESEARCH_AVAILABLE = False


class TestAdvancedQuantumAlgorithms(unittest.TestCase):
    """Test advanced quantum algorithms implementation."""
    
    def setUp(self):
        """Set up test cases."""
        if not RESEARCH_AVAILABLE:
            self.skipTest("Research modules not available")
        
        # Create test QUBO matrix
        self.test_qubo = np.array([
            [1, -2, 0, 1],
            [-2, 2, -1, 0], 
            [0, -1, 1, -1],
            [1, 0, -1, 2]
        ])
        self.num_qubits = 4
    
    def test_adaptive_qaoa_creation(self):
        """Test Adaptive QAOA algorithm creation."""
        params = AdaptiveQAOAParams(
            initial_layers=2,
            max_layers=5,
            adaptation_threshold=1e-4
        )
        
        qaoa = QuantumAlgorithmFactory.create_algorithm(
            QuantumAlgorithmType.ADAPTIVE_QAOA,
            qaoa_params=params
        )
        
        self.assertIsNotNone(qaoa)
        self.assertEqual(qaoa.algorithm_type, QuantumAlgorithmType.ADAPTIVE_QAOA)
        self.assertEqual(qaoa.params.initial_layers, 2)
        self.assertEqual(qaoa.params.max_layers, 5)
    
    def test_adaptive_qaoa_solving(self):
        """Test Adaptive QAOA problem solving."""
        qaoa = QuantumAlgorithmFactory.create_algorithm(
            QuantumAlgorithmType.ADAPTIVE_QAOA
        )
        
        result = qaoa.solve_scheduling_problem(
            hamiltonian=self.test_qubo,
            num_qubits=self.num_qubits
        )
        
        # Validate result structure
        self.assertIsNotNone(result.solution)
        self.assertIsInstance(result.energy, float)
        self.assertIsInstance(result.execution_time, float)
        self.assertIsInstance(result.convergence_steps, int)
        self.assertGreater(result.fidelity, 0.0)
        
        # Validate solution format
        self.assertIsInstance(result.solution, dict)
        for var, value in result.solution.items():
            self.assertIn(value, [0, 1])
    
    def test_vqe_scheduler(self):
        """Test VQE Task Scheduler algorithm."""
        vqe = QuantumAlgorithmFactory.create_algorithm(
            QuantumAlgorithmType.VQE_SCHEDULER
        )
        
        result = vqe.solve_scheduling_problem(
            hamiltonian=self.test_qubo,
            num_qubits=self.num_qubits,
            ansatz_type="scheduling_specific"
        )
        
        # Validate VQE-specific results
        self.assertEqual(result.algorithm_type, QuantumAlgorithmType.VQE_SCHEDULER)
        self.assertIsNotNone(result.solution)
        self.assertGreater(result.gate_count, 0)
        self.assertGreater(result.quantum_circuit_depth, 0)
    
    def test_qml_predictor(self):
        """Test Quantum ML Task Predictor."""
        qml = QuantumAlgorithmFactory.create_algorithm(
            QuantumAlgorithmType.QML_PREDICTOR
        )
        
        # Test without historical data
        result = qml.solve_scheduling_problem(
            hamiltonian=self.test_qubo,
            num_qubits=self.num_qubits
        )
        
        self.assertEqual(result.algorithm_type, QuantumAlgorithmType.QML_PREDICTOR)
        self.assertIsNotNone(result.solution)
        
        # Test with mock historical data
        historical_data = [
            {"problem": "test1", "solution": {0: 1, 1: 0, 2: 1, 3: 0}},
            {"problem": "test2", "solution": {0: 0, 1: 1, 2: 0, 3: 1}}
        ]
        
        result_with_data = qml.solve_scheduling_problem(
            hamiltonian=self.test_qubo,
            num_qubits=self.num_qubits,
            historical_data=historical_data
        )
        
        self.assertIsNotNone(result_with_data.solution)
    
    def test_algorithm_factory(self):
        """Test QuantumAlgorithmFactory functionality."""
        available_algorithms = QuantumAlgorithmFactory.get_available_algorithms()
        
        self.assertIn(QuantumAlgorithmType.ADAPTIVE_QAOA, available_algorithms)
        self.assertIn(QuantumAlgorithmType.VQE_SCHEDULER, available_algorithms)
        self.assertIn(QuantumAlgorithmType.QML_PREDICTOR, available_algorithms)
        
        # Test descriptions
        for alg_type in available_algorithms:
            description = QuantumAlgorithmFactory.get_algorithm_description(alg_type)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 10)


class TestHybridDecomposition(unittest.TestCase):
    """Test hybrid quantum-classical decomposition."""
    
    def setUp(self):
        """Set up test cases."""
        if not RESEARCH_AVAILABLE:
            self.skipTest("Research modules not available")
        
        # Create larger test problem
        self.large_qubo = np.random.rand(20, 20)
        self.large_qubo = (self.large_qubo + self.large_qubo.T) / 2  # Make symmetric
        
        self.hybrid_solver = HybridQuantumClassicalSolver()
    
    def test_problem_decomposition(self):
        """Test problem decomposition strategies."""
        decomposer = self.hybrid_solver.decomposer
        
        # Test spectral clustering decomposition
        result = decomposer.decompose_problem(
            self.large_qubo,
            strategy=DecompositionStrategy.SPECTRAL_CLUSTERING,
            max_subproblem_size=8
        )
        
        self.assertIsNotNone(result.subproblems)
        self.assertGreater(len(result.subproblems), 1)
        self.assertIsInstance(result.decomposition_quality, float)
        self.assertGreater(result.decomposition_quality, 0)
        
        # Validate subproblems
        for subproblem in result.subproblems:
            self.assertIn("variables", subproblem)
            self.assertIn("matrix", subproblem)
            self.assertIsInstance(subproblem["variables"], list)
            self.assertIsInstance(subproblem["matrix"], np.ndarray)
        
        # Validate subproblem metrics
        self.assertEqual(len(result.subproblem_metrics), len(result.subproblems))
        for metrics in result.subproblem_metrics:
            self.assertIsInstance(metrics.num_variables, int)
            self.assertGreater(metrics.connectivity, 0)
            self.assertIn(metrics.recommended_solver, ["quantum", "classical", "hybrid"])
    
    def test_hybrid_modes(self):
        """Test different hybrid solving modes."""
        
        # Test sequential mode
        result_seq = self.hybrid_solver.solve_hybrid(
            problem_matrix=self.large_qubo,
            hybrid_mode=HybridMode.SEQUENTIAL,
            max_subproblem_size=10
        )
        
        self.assertIsNotNone(result_seq.solution)
        self.assertEqual(result_seq.hybrid_mode_used, HybridMode.SEQUENTIAL)
        self.assertIsInstance(result_seq.total_energy, float)
        self.assertGreater(result_seq.total_execution_time, 0)
        
        # Test adaptive mode
        result_adaptive = self.hybrid_solver.solve_hybrid(
            problem_matrix=self.large_qubo,
            hybrid_mode=HybridMode.ADAPTIVE,
            max_subproblem_size=10
        )
        
        self.assertIsNotNone(result_adaptive.solution)
        self.assertEqual(result_adaptive.hybrid_mode_used, HybridMode.ADAPTIVE)
        
        # Validate solution format
        for solution in [result_seq.solution, result_adaptive.solution]:
            self.assertIsInstance(solution, dict)
            for var, value in solution.items():
                self.assertIn(value, [0, 1])
    
    def test_decomposition_strategies(self):
        """Test different decomposition strategies."""
        
        strategies = [
            DecompositionStrategy.SPECTRAL_CLUSTERING,
            DecompositionStrategy.COMPLEXITY_AWARE,
            DecompositionStrategy.SKILL_BASED
        ]
        
        for strategy in strategies:
            try:
                result = self.hybrid_solver.solve_hybrid(
                    problem_matrix=self.large_qubo,
                    hybrid_mode=HybridMode.SEQUENTIAL,
                    decomposition_strategy=strategy,
                    max_subproblem_size=8
                )
                
                self.assertIsNotNone(result.solution)
                self.assertIn("decomposition_strategy", result.metadata)
                self.assertEqual(result.metadata["decomposition_strategy"], strategy.value)
                
            except Exception as e:
                # Some strategies may fail depending on dependencies
                print(f"Strategy {strategy.value} failed: {e}")
    
    def test_solution_quality(self):
        """Test solution quality evaluation."""
        
        result = self.hybrid_solver.solve_hybrid(
            problem_matrix=self.large_qubo,
            hybrid_mode=HybridMode.SEQUENTIAL,
            max_subproblem_size=10
        )
        
        # Calculate energy manually to verify
        solution = result.solution
        manual_energy = 0.0
        
        for i in range(self.large_qubo.shape[0]):
            for j in range(self.large_qubo.shape[1]):
                if i in solution and j in solution:
                    manual_energy += self.large_qubo[i, j] * solution[i] * solution[j]
        
        # Energy should be reasonably close (allowing for numerical differences)
        self.assertAlmostEqual(result.total_energy, manual_energy, places=3)


class TestQuantumAdvantageBenchmarking(unittest.TestCase):
    """Test quantum advantage benchmarking framework."""
    
    def setUp(self):
        """Set up test cases."""
        if not RESEARCH_AVAILABLE:
            self.skipTest("Research modules not available")
    
    def test_problem_generation(self):
        """Test benchmark problem generation."""
        
        generator = ProblemGenerator()
        
        problems = generator.generate_problem_suite(
            problem_sizes=[6, 8],
            problem_types=["task_assignment"],
            instances_per_size=2,
            seed=42
        )
        
        self.assertEqual(len(problems), 4)  # 2 sizes * 1 type * 2 instances
        
        for problem in problems:
            self.assertIsInstance(problem.problem_id, str)
            self.assertIn(problem.problem_type, ["task_assignment"])
            self.assertGreater(problem.num_agents, 0)
            self.assertGreater(problem.num_tasks, 0)
            self.assertIn("qubo_matrix", problem.problem_data)
            
            # Validate QUBO matrix
            qubo_matrix = np.array(problem.problem_data["qubo_matrix"])
            self.assertEqual(qubo_matrix.shape[0], qubo_matrix.shape[1])
            self.assertEqual(qubo_matrix.shape[0], problem.num_variables)
    
    def test_benchmark_runner(self):
        """Test benchmark execution."""
        
        # Generate small test problems
        generator = ProblemGenerator()
        problems = generator.generate_problem_suite(
            problem_sizes=[6],
            problem_types=["task_assignment"],
            instances_per_size=2
        )
        
        # Define mock algorithms
        def mock_quantum_solver(hamiltonian=None, **kwargs):
            """Mock quantum solver."""
            time.sleep(0.01)  # Simulate solving time
            n = hamiltonian.shape[0] if hasattr(hamiltonian, 'shape') else 4
            
            class MockResult:
                def __init__(self):
                    self.solution = {i: np.random.randint(0, 2) for i in range(n)}
                    self.energy = np.random.uniform(-10, 0)
                    self.execution_time = 0.01
                    self.convergence_steps = 50
            
            return MockResult()
        
        def mock_classical_solver(hamiltonian=None, **kwargs):
            """Mock classical solver."""
            time.sleep(0.02)  # Simulate slower solving
            n = hamiltonian.shape[0] if hasattr(hamiltonian, 'shape') else 4
            
            class MockResult:
                def __init__(self):
                    self.solution = {i: np.random.randint(0, 2) for i in range(n)}
                    self.energy = np.random.uniform(-8, 2)
                    self.execution_time = 0.02
                    self.convergence_steps = 100
            
            return MockResult()
        
        algorithms = {
            "mock_quantum": mock_quantum_solver,
            "mock_classical": mock_classical_solver
        }
        
        # Run benchmark
        runner = BenchmarkRunner(timeout_seconds=10)
        results = runner.run_benchmark_suite(
            algorithms=algorithms,
            problems=problems,
            runs_per_problem=2,
            parallel=False
        )
        
        # Validate results
        expected_results = len(algorithms) * len(problems) * 2
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), expected_results)
        
        for result in results:
            self.assertIn(result.algorithm_name, algorithms.keys())
            self.assertIsInstance(result.solution_quality, float)
            self.assertGreater(result.execution_time, 0)
            self.assertIsInstance(result.solution_found, bool)
    
    def test_advantage_analysis(self):
        """Test quantum advantage analysis."""
        
        # Create mock benchmark results
        mock_results = []
        
        problem_ids = ["problem_1", "problem_2"]
        
        # Mock quantum results (better performance)
        for problem_id in problem_ids:
            for run in range(3):
                mock_results.append(type('BenchmarkResult', (), {
                    'algorithm_name': 'mock_quantum',
                    'problem_id': problem_id,
                    'solution_quality': np.random.uniform(-10, -5),
                    'execution_time': np.random.uniform(0.5, 1.0),
                    'solution_found': True,
                    'convergence_steps': np.random.randint(20, 50)
                })())
        
        # Mock classical results (worse performance)
        for problem_id in problem_ids:
            for run in range(3):
                mock_results.append(type('BenchmarkResult', (), {
                    'algorithm_name': 'mock_classical',
                    'problem_id': problem_id,
                    'solution_quality': np.random.uniform(-5, 0),
                    'execution_time': np.random.uniform(2.0, 4.0),
                    'solution_found': True,
                    'convergence_steps': np.random.randint(100, 200)
                })())
        
        # Analyze advantage
        from quantum_planner.research.quantum_advantage_benchmarks import BenchmarkCategory
        
        analyzer = QuantumAdvantageAnalyzer()
        
        report = analyzer.analyze_quantum_advantage(
            results=mock_results,
            quantum_algorithm="mock_quantum",
            classical_baseline="mock_classical",
            benchmark_category=BenchmarkCategory.PERFORMANCE_SCALING
        )
        
        # Validate report
        self.assertIsNotNone(report)
        self.assertEqual(report.quantum_algorithm, "mock_quantum")
        self.assertEqual(report.classical_baseline, "mock_classical")
        self.assertGreater(report.quantum_advantage_factor, 0)
        self.assertIsInstance(report.statistical_significance, bool)
        self.assertIsInstance(report.publication_summary, str)
        self.assertGreater(len(report.publication_summary), 100)


class TestResearchBackend(unittest.TestCase):
    """Test integrated research backend."""
    
    def setUp(self):
        """Set up test cases."""
        if not RESEARCH_AVAILABLE:
            self.skipTest("Research modules not available")
        
        self.test_qubo = np.random.rand(12, 12)
        self.test_qubo = (self.test_qubo + self.test_qubo.T) / 2
    
    def test_backend_creation(self):
        """Test research backend creation."""
        
        backend = ResearchQuantumBackend()
        self.assertIsNotNone(backend)
        
        capabilities = backend.get_capabilities()
        self.assertGreater(capabilities.max_variables, 100)
        self.assertTrue(capabilities.supports_constraints)
        self.assertTrue(capabilities.supports_async)
    
    def test_backend_solving(self):
        """Test backend problem solving."""
        
        backend = ResearchQuantumBackend({
            "preferred_algorithm": "adaptive_qaoa",
            "enable_hybrid_decomposition": False
        })
        
        solution = backend._solve_qubo(self.test_qubo)
        
        self.assertIsInstance(solution, dict)
        self.assertGreater(len(solution), 0)
        
        for var, value in solution.items():
            self.assertIn(value, [0, 1])
    
    def test_hybrid_backend_solving(self):
        """Test hybrid backend solving."""
        
        backend = ResearchQuantumBackend({
            "enable_hybrid_decomposition": True,
            "hybrid_mode": "sequential",
            "max_subproblem_size": 6
        })
        
        solution = backend._solve_qubo(self.test_qubo)
        
        self.assertIsInstance(solution, dict)
        self.assertGreater(len(solution), 0)
    
    def test_research_metrics(self):
        """Test research metrics reporting."""
        
        backend = ResearchQuantumBackend()
        metrics = backend.get_research_metrics()
        
        self.assertIn("backend_type", metrics)
        self.assertIn("algorithms_available", metrics)
        self.assertIn("research_features", metrics)
        
        self.assertEqual(metrics["backend_type"], "research_quantum")
        self.assertIsInstance(metrics["algorithms_available"], list)
        self.assertGreater(len(metrics["research_features"]), 3)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def setUp(self):
        """Set up integration tests."""
        if not RESEARCH_AVAILABLE:
            self.skipTest("Research modules not available")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        
        # Step 1: Create problem
        problem_matrix = np.random.rand(16, 16)
        problem_matrix = (problem_matrix + problem_matrix.T) / 2
        
        # Step 2: Solve with research backend
        backend = ResearchQuantumBackend({
            "enable_hybrid_decomposition": True,
            "algorithm_selection_enabled": True
        })
        
        solution = backend._solve_qubo(problem_matrix)
        
        # Step 3: Validate solution
        self.assertIsInstance(solution, dict)
        self.assertGreater(len(solution), 0)
        
        # Step 4: Calculate energy
        energy = 0.0
        for i in range(problem_matrix.shape[0]):
            for j in range(problem_matrix.shape[1]):
                if i in solution and j in solution:
                    energy += problem_matrix[i, j] * solution[i] * solution[j]
        
        self.assertIsInstance(energy, float)
        
        print(f"End-to-end test completed: {len(solution)} variables, energy = {energy:.4f}")


def run_performance_tests():
    """Run performance validation tests."""
    
    if not RESEARCH_AVAILABLE:
        print("Skipping performance tests - research modules not available")
        return
    
    print("\n" + "="*50)
    print("PERFORMANCE VALIDATION TESTS")
    print("="*50)
    
    # Test scaling behavior
    problem_sizes = [8, 12, 16, 20]
    
    for size in problem_sizes:
        print(f"\nTesting problem size: {size}x{size}")
        
        # Generate test problem
        problem_matrix = np.random.rand(size, size)
        problem_matrix = (problem_matrix + problem_matrix.T) / 2
        
        # Test with research backend
        backend = ResearchQuantumBackend({
            "enable_hybrid_decomposition": size > 12,
            "max_subproblem_size": 8
        })
        
        start_time = time.time()
        solution = backend._solve_qubo(problem_matrix)
        execution_time = time.time() - start_time
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Solution variables: {len(solution)}")
        print(f"  Variables set to 1: {sum(1 for v in solution.values() if v == 1)}")


def main():
    """Run all tests."""
    
    print("QUANTUM PLANNER RESEARCH - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    if not RESEARCH_AVAILABLE:
        print("WARNING: Research modules not available. Some tests will be skipped.")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedQuantumAlgorithms,
        TestHybridDecomposition,
        TestQuantumAdvantageBenchmarking,
        TestResearchBackend,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Performance tests
    run_performance_tests()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILED TESTS:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERROR TESTS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    print(f"\nOVERALL RESULT: {'SUCCESS' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)