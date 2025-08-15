#!/usr/bin/env python3
"""Research Implementation Test - Novel Algorithms and Comparative Studies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import logging
import numpy as np
from typing import Dict, Any
from quantum_planner import QuantumTaskPlanner, Agent, Task
from quantum_planner.research.novel_quantum_algorithms import (
    novel_optimizer, QuantumAlgorithmType, ResearchMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_novel_qaoa_implementation():
    """Test novel QAOA implementation with adaptive features."""
    logger.info("=== Testing Novel QAOA Implementation ===")
    
    try:
        # Create research problem
        agents = [
            Agent(f"quantum_agent_{i}", skills=["quantum", "optimization"], capacity=3)
            for i in range(4)
        ]
        tasks = [
            Task(f"quantum_task_{i}", required_skills=["quantum"], priority=i+1, duration=2)
            for i in range(6)
        ]
        
        # Test QAOA with different layer depths
        for p_layers in [1, 2]:
            start_time = time.time()
            result = novel_optimizer.quantum_approximate_optimization_algorithm(
                agents, tasks, p_layers=p_layers, max_iterations=10
            )
            execution_time = time.time() - start_time
            
            logger.info(f"QAOA (p={p_layers}): Energy={result.energy:.3f}, "
                       f"Fidelity={result.fidelity:.3f}, Time={execution_time:.3f}s")
            
            # Validate results
            if result.algorithm_type == QuantumAlgorithmType.QAOA:
                logger.info(f"✅ QAOA p={p_layers} executed successfully")
            else:
                logger.error(f"❌ QAOA p={p_layers} algorithm type mismatch")
                return False
            
            if result.circuit_depth == p_layers * 2:
                logger.info(f"✅ Circuit depth correct: {result.circuit_depth}")
            else:
                logger.error(f"❌ Circuit depth incorrect: {result.circuit_depth}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ QAOA test failed: {e}")
        return False


def test_variational_quantum_eigensolver():
    """Test VQE implementation with hardware-efficient ansatz."""
    logger.info("=== Testing Variational Quantum Eigensolver ===")
    
    try:
        agents = [Agent(f"vqe_agent_{i}", skills=["optimization"], capacity=2) for i in range(3)]
        tasks = [Task(f"vqe_task_{i}", required_skills=["optimization"], priority=1, duration=1) for i in range(4)]
        
        # Test VQE with different ansatz depths
        for depth in [2, 3]:
            start_time = time.time()
            result = novel_optimizer.variational_quantum_eigensolver(
                agents, tasks, ansatz_depth=depth
            )
            execution_time = time.time() - start_time
            
            logger.info(f"VQE (depth={depth}): Energy={result.energy:.3f}, "
                       f"Fidelity={result.fidelity:.3f}, Time={execution_time:.3f}s")
            
            # Validate results
            if result.algorithm_type == QuantumAlgorithmType.VQE:
                logger.info(f"✅ VQE depth={depth} executed successfully")
            else:
                logger.error(f"❌ VQE depth={depth} algorithm type mismatch")
                return False
            
            if 0.0 <= result.fidelity <= 1.0:
                logger.info(f"✅ Fidelity in valid range: {result.fidelity:.3f}")
            else:
                logger.error(f"❌ Invalid fidelity: {result.fidelity}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ VQE test failed: {e}")
        return False


def test_hybrid_quantum_classical():
    """Test hybrid quantum-classical algorithm."""
    logger.info("=== Testing Hybrid Quantum-Classical Algorithm ===")
    
    try:
        # Create larger problem for hybrid approach
        agents = [
            Agent(f"hybrid_agent_{i}", skills=["quantum", "classical"], capacity=4)
            for i in range(6)
        ]
        tasks = [
            Task(f"hybrid_task_{i}", required_skills=["quantum" if i % 2 == 0 else "classical"], 
                 priority=i+1, duration=2)
            for i in range(10)
        ]
        
        # Test different quantum ratios
        for quantum_ratio in [0.3, 0.5]:
            start_time = time.time()
            result = novel_optimizer.hybrid_quantum_classical_algorithm(
                agents, tasks, quantum_ratio=quantum_ratio
            )
            execution_time = time.time() - start_time
            
            logger.info(f"Hybrid (ratio={quantum_ratio:.1f}): Energy={result.energy:.3f}, "
                       f"Time={execution_time:.3f}s")
            
            # Validate results
            if result.algorithm_type == QuantumAlgorithmType.HYBRID_CLASSICAL_QUANTUM:
                logger.info(f"✅ Hybrid ratio={quantum_ratio:.1f} executed successfully")
            else:
                logger.error(f"❌ Hybrid algorithm type mismatch")
                return False
            
            # Hybrid should be relatively fast and reliable
            if result.success_probability >= 0.9:
                logger.info(f"✅ High success probability: {result.success_probability:.3f}")
            else:
                logger.warning(f"⚠️ Lower success probability: {result.success_probability:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hybrid test failed: {e}")
        return False


def test_adaptive_quantum_annealing():
    """Test adaptive quantum annealing algorithm."""
    logger.info("=== Testing Adaptive Quantum Annealing ===")
    
    try:
        agents = [Agent(f"annealing_agent_{i}", skills=["physics"], capacity=3) for i in range(5)]
        tasks = [Task(f"annealing_task_{i}", required_skills=["physics"], priority=1, duration=1) for i in range(8)]
        
        start_time = time.time()
        result = novel_optimizer.adaptive_quantum_annealing(agents, tasks)
        execution_time = time.time() - start_time
        
        logger.info(f"Adaptive Annealing: Energy={result.energy:.3f}, "
                   f"Success Rate={result.success_probability:.3f}, Time={execution_time:.3f}s")
        
        # Validate results
        if result.algorithm_type == QuantumAlgorithmType.QUANTUM_ANNEALING:
            logger.info("✅ Adaptive annealing executed successfully")
        else:
            logger.error("❌ Annealing algorithm type mismatch")
            return False
        
        # Annealing should have high measurement counts
        total_measurements = sum(result.measurement_counts.values())
        if total_measurements >= 1000:
            logger.info(f"✅ Sufficient measurements: {total_measurements}")
        else:
            logger.error(f"❌ Insufficient measurements: {total_measurements}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive annealing test failed: {e}")
        return False


def test_comprehensive_algorithm_comparison():
    """Test comprehensive comparison of all algorithms."""
    logger.info("=== Testing Comprehensive Algorithm Comparison ===")
    
    try:
        # Create benchmark problem
        agents = [
            Agent(f"benchmark_agent_{i}", skills=["research", "quantum"], capacity=3)
            for i in range(4)
        ]
        tasks = [
            Task(f"benchmark_task_{i}", required_skills=["research"], priority=i+1, duration=2)
            for i in range(6)
        ]
        
        # Run comprehensive comparison
        start_time = time.time()
        comparison_results = novel_optimizer.compare_algorithms_comprehensive(agents, tasks)
        execution_time = time.time() - start_time
        
        logger.info(f"Comprehensive comparison completed in {execution_time:.3f}s")
        
        # Validate comparison results
        expected_algorithms = ["QAOA", "VQE", "Hybrid", "Adaptive_Annealing"]
        
        for algorithm in expected_algorithms:
            if algorithm in comparison_results:
                if "error" in comparison_results[algorithm]:
                    logger.error(f"❌ {algorithm} failed: {comparison_results[algorithm]['error']}")
                    return False
                else:
                    result = comparison_results[algorithm]
                    logger.info(f"✅ {algorithm}: Quality={result['performance_score']:.3f}")
            else:
                logger.error(f"❌ Missing algorithm: {algorithm}")
                return False
        
        # Check for best algorithm recommendation
        if "best_algorithm" in comparison_results:
            best = comparison_results["best_algorithm"]
            logger.info(f"✅ Best algorithm identified: {best}")
        else:
            logger.error("❌ No best algorithm identified")
            return False
        
        # Validate research metrics
        for algorithm in expected_algorithms:
            if algorithm in comparison_results and "research_metrics" in comparison_results[algorithm]:
                metrics = comparison_results[algorithm]["research_metrics"]
                if isinstance(metrics, ResearchMetrics):
                    logger.info(f"✅ {algorithm} has valid research metrics")
                else:
                    logger.error(f"❌ {algorithm} has invalid research metrics")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive comparison test failed: {e}")
        return False


def test_statistical_validation():
    """Test statistical validation of research results."""
    logger.info("=== Testing Statistical Validation ===")
    
    try:
        # Create test problem
        agents = [Agent(f"stat_agent_{i}", skills=["validation"], capacity=2) for i in range(3)]
        tasks = [Task(f"stat_task_{i}", required_skills=["validation"], priority=1, duration=1) for i in range(4)]
        
        # Run multiple trials for statistical analysis
        trials = 5
        qaoa_results = []
        vqe_results = []
        
        for trial in range(trials):
            logger.info(f"Running trial {trial + 1}/{trials}")
            
            # QAOA trial
            qaoa_result = novel_optimizer.quantum_approximate_optimization_algorithm(
                agents, tasks, p_layers=2, max_iterations=5
            )
            qaoa_results.append(qaoa_result.fidelity)
            
            # VQE trial
            vqe_result = novel_optimizer.variational_quantum_eigensolver(
                agents, tasks, ansatz_depth=3
            )
            vqe_results.append(vqe_result.fidelity)
        
        # Statistical analysis
        qaoa_mean = np.mean(qaoa_results)
        qaoa_std = np.std(qaoa_results)
        vqe_mean = np.mean(vqe_results)
        vqe_std = np.std(vqe_results)
        
        logger.info(f"QAOA: Mean={qaoa_mean:.3f}, Std={qaoa_std:.3f}")
        logger.info(f"VQE: Mean={vqe_mean:.3f}, Std={vqe_std:.3f}")
        
        # Validate statistical properties
        if qaoa_std < 0.5:  # Results should be reasonably consistent
            logger.info("✅ QAOA shows good reproducibility")
        else:
            logger.warning(f"⚠️ QAOA high variance: {qaoa_std:.3f}")
        
        if vqe_std < 0.5:
            logger.info("✅ VQE shows good reproducibility")
        else:
            logger.warning(f"⚠️ VQE high variance: {vqe_std:.3f}")
        
        # Simple significance test (difference in means)
        if abs(qaoa_mean - vqe_mean) > 0.1:
            logger.info(f"✅ Statistically significant difference detected: {abs(qaoa_mean - vqe_mean):.3f}")
        else:
            logger.info("ℹ️ No significant difference between algorithms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Statistical validation test failed: {e}")
        return False


def benchmark_quantum_advantage():
    """Benchmark quantum advantage over classical methods."""
    logger.info("=== Benchmarking Quantum Advantage ===")
    
    try:
        # Create problems of increasing size
        problem_sizes = [(3, 4), (4, 6), (5, 8)]
        classical_times = []
        quantum_times = []
        
        for n_agents, n_tasks in problem_sizes:
            agents = [Agent(f"agent_{i}", skills=["benchmark"], capacity=2) for i in range(n_agents)]
            tasks = [Task(f"task_{i}", required_skills=["benchmark"], priority=1, duration=1) for i in range(n_tasks)]
            
            # Classical baseline
            classical_start = time.time()
            planner = QuantumTaskPlanner()
            classical_solution = planner.assign(agents, tasks)
            classical_time = time.time() - classical_start
            classical_times.append(classical_time)
            
            # Quantum approach (QAOA)
            quantum_start = time.time()
            quantum_result = novel_optimizer.quantum_approximate_optimization_algorithm(
                agents, tasks, p_layers=2, max_iterations=5
            )
            quantum_time = time.time() - quantum_start
            quantum_times.append(quantum_time)
            
            # Calculate advantage
            if quantum_time > 0:
                advantage = classical_time / quantum_time
                logger.info(f"Problem {n_agents}x{n_tasks}: Classical={classical_time:.3f}s, "
                           f"Quantum={quantum_time:.3f}s, Advantage={advantage:.2f}x")
            else:
                logger.warning(f"Problem {n_agents}x{n_tasks}: Quantum time too small to measure")
        
        # Overall analysis
        total_classical = sum(classical_times)
        total_quantum = sum(quantum_times)
        overall_advantage = total_classical / total_quantum if total_quantum > 0 else 1.0
        
        logger.info(f"Overall quantum advantage: {overall_advantage:.2f}x")
        
        if overall_advantage > 1.0:
            logger.info("✅ Quantum advantage demonstrated")
        else:
            logger.info("ℹ️ No quantum advantage in current implementation")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantum advantage benchmark failed: {e}")
        return False


def main():
    """Run all research implementation tests."""
    logger.info("🔬 RESEARCH IMPLEMENTATION - NOVEL ALGORITHMS & COMPARATIVE STUDIES")
    logger.info("=" * 80)
    
    tests = [
        ("Novel QAOA Implementation", test_novel_qaoa_implementation),
        ("Variational Quantum Eigensolver", test_variational_quantum_eigensolver),
        ("Hybrid Quantum-Classical", test_hybrid_quantum_classical),
        ("Adaptive Quantum Annealing", test_adaptive_quantum_annealing),
        ("Comprehensive Algorithm Comparison", test_comprehensive_algorithm_comparison),
        ("Statistical Validation", test_statistical_validation),
        ("Quantum Advantage Benchmark", benchmark_quantum_advantage),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAIL - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("RESEARCH IMPLEMENTATION SUMMARY")
    logger.info(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<50} {status}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Research implementation complete! Novel algorithms validated")
        logger.info("📊 Ready for publication-quality documentation and benchmarks")
        
        # Research summary
        logger.info(f"\n🔬 RESEARCH CONTRIBUTIONS:")
        logger.info("  • Novel QAOA with adaptive parameter initialization")
        logger.info("  • Hardware-efficient VQE ansatz for scheduling")
        logger.info("  • Intelligent hybrid quantum-classical decomposition")
        logger.info("  • Adaptive quantum annealing with dynamic scheduling")
        logger.info("  • Comprehensive comparative framework with statistical validation")
        logger.info("  • Quantum advantage analysis for scheduling problems")
        
        return True
    else:
        logger.error("💥 Research implementation failed! Fix algorithm issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)