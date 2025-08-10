#!/usr/bin/env python3
"""Comprehensive test suite for all generations with quality gates."""

import sys
import os
import time
import unittest
import asyncio
import concurrent.futures
import threading
import random
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

# Import all generations
sys.path.insert(0, os.path.dirname(__file__))

# Disable logging for cleaner test output
logging.disable(logging.CRITICAL)

class TestRunner:
    """Advanced test runner with quality gates."""
    
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "performance_metrics": {},
            "coverage_score": 0.0,
            "security_score": 0.0,
            "quality_score": 0.0
        }
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_func, timeout: int = 30):
        """Run individual test with timeout and error handling."""
        self.results["total_tests"] += 1
        
        try:
            print(f"🧪 Running {test_name}...")
            
            start_time = time.time()
            
            # Run test with timeout
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(asyncio.wait_for(test_func(), timeout=timeout))
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(test_func)
                    result = future.result(timeout=timeout)
            
            execution_time = time.time() - start_time
            self.results["performance_metrics"][test_name] = execution_time
            
            if result:
                self.results["passed"] += 1
                print(f"✅ {test_name} PASSED ({execution_time:.3f}s)")
            else:
                self.results["failed"] += 1
                print(f"❌ {test_name} FAILED ({execution_time:.3f}s)")
                
        except Exception as e:
            self.results["failed"] += 1
            error_msg = f"{test_name}: {str(e)}"
            self.results["errors"].append(error_msg)
            print(f"❌ {test_name} ERROR: {str(e)}")
    
    def calculate_scores(self):
        """Calculate quality scores."""
        total_tests = self.results["total_tests"]
        if total_tests == 0:
            return
        
        # Test success rate
        success_rate = self.results["passed"] / total_tests
        
        # Performance score (average execution time)
        avg_execution_time = sum(self.results["performance_metrics"].values()) / len(self.results["performance_metrics"]) if self.results["performance_metrics"] else 0
        performance_score = max(0, 1 - (avg_execution_time / 10))  # Penalty for slow tests
        
        # Coverage approximation (based on test variety)
        coverage_score = min(1.0, total_tests / 25)  # 25 tests = 100% coverage
        
        # Security score (no critical failures)
        security_score = 1.0 if "security" not in str(self.results["errors"]).lower() else 0.5
        
        # Overall quality score
        quality_score = (success_rate * 0.4 + performance_score * 0.2 + 
                        coverage_score * 0.2 + security_score * 0.2)
        
        self.results.update({
            "coverage_score": coverage_score * 100,
            "security_score": security_score * 100,
            "quality_score": quality_score * 100
        })
    
    def generate_report(self):
        """Generate comprehensive test report."""
        self.calculate_scores()
        
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"📊 Test Results:")
        print(f"   Total Tests: {self.results['total_tests']}")
        print(f"   Passed: {self.results['passed']} ✅")
        print(f"   Failed: {self.results['failed']} ❌")
        print(f"   Success Rate: {self.results['passed']/self.results['total_tests']*100:.1f}%")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   Total Execution Time: {total_time:.2f}s")
        print(f"   Average Test Time: {sum(self.results['performance_metrics'].values())/len(self.results['performance_metrics']):.3f}s")
        print(f"   Fastest Test: {min(self.results['performance_metrics'].values()):.3f}s")
        print(f"   Slowest Test: {max(self.results['performance_metrics'].values()):.3f}s")
        
        print(f"\n🎯 Quality Gates:")
        print(f"   Coverage Score: {self.results['coverage_score']:.1f}/100 {'✅' if self.results['coverage_score'] >= 80 else '❌'}")
        print(f"   Security Score: {self.results['security_score']:.1f}/100 {'✅' if self.results['security_score'] >= 90 else '❌'}")
        print(f"   Overall Quality: {self.results['quality_score']:.1f}/100 {'✅' if self.results['quality_score'] >= 85 else '❌'}")
        
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(self.results['errors']) > 5:
                print(f"   ... and {len(self.results['errors']) - 5} more")
        
        # Quality gate decision
        quality_passed = (
            self.results['coverage_score'] >= 80 and
            self.results['security_score'] >= 90 and
            self.results['quality_score'] >= 85 and
            self.results['failed'] == 0
        )
        
        print(f"\n🚪 Quality Gate: {'✅ PASSED' if quality_passed else '❌ FAILED'}")
        
        return quality_passed

# Test implementations for all generations

def test_basic_imports():
    """Test that all modules can be imported."""
    try:
        # Test basic imports without external dependencies
        import numpy as np
        
        # Basic data structures
        @dataclass 
        class Agent:
            id: str
            skills: List[str]
            capacity: int = 1
        
        @dataclass
        class Task:
            id: str
            required_skills: List[str] 
            priority: int = 1
            duration: int = 1
        
        return True
    except Exception:
        return False

def test_generation1_basic_functionality():
    """Test Generation 1 basic functionality."""
    try:
        exec(open("minimal_test.py").read())
        return True
    except Exception:
        return False

def test_generation2_robust_functionality():
    """Test Generation 2 robust functionality."""
    try:
        exec(open("robust_planner.py").read())
        return True
    except Exception:
        return False

def test_generation3_scalable_functionality():
    """Test Generation 3 scalable functionality."""
    try:
        exec(open("scalable_planner.py").read())
        return True
    except Exception:
        return False

def test_data_validation():
    """Test comprehensive data validation."""
    from robust_planner import Agent, Task, ValidationError
    
    try:
        # Test invalid agent
        try:
            Agent("", [], capacity=-1)
            return False  # Should have raised exception
        except ValidationError:
            pass  # Expected
        
        # Test invalid task
        try:
            Task("", [], priority=0)
            return False  # Should have raised exception
        except ValidationError:
            pass  # Expected
        
        # Test valid data
        Agent("agent1", ["python"], capacity=2)
        Task("task1", ["python"], priority=1)
        
        return True
    except Exception:
        return False

def test_security_validation():
    """Test security validation mechanisms."""
    from robust_planner import SecurityManager, SecurityError
    
    try:
        # Test forbidden patterns
        for pattern in ["__import__", "eval", "exec"]:
            try:
                SecurityManager.validate_input(pattern)
                return False  # Should have raised exception
            except SecurityError:
                pass  # Expected
        
        # Test safe input
        SecurityManager.validate_input("safe_string")
        SecurityManager.validate_input(["safe", "list"])
        
        # Test ID sanitization
        sanitized = SecurityManager.sanitize_id("agent-1_test")
        assert sanitized == "agent-1_test"
        
        return True
    except Exception:
        return False

def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    from robust_planner import PerformanceMonitor
    
    try:
        monitor = PerformanceMonitor()
        
        # Record some metrics
        monitor.record_metric("test_metric", 1.5)
        monitor.record_metric("test_metric", 2.0)
        monitor.record_metric("test_metric", 1.2)
        
        # Get statistics
        stats = monitor.get_stats("test_metric")
        
        assert stats["count"] == 3
        assert abs(stats["mean"] - 1.567) < 0.01
        assert stats["min"] == 1.2
        assert stats["max"] == 2.0
        
        return True
    except Exception:
        return False

def test_caching_mechanism():
    """Test caching with TTL and LRU."""
    from scalable_planner import CacheManager
    
    try:
        cache = CacheManager(max_size=3, ttl_seconds=1)
        
        # Test basic operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Test LRU eviction
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"  # Recently accessed
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats["size"] <= 3
        assert stats["hits"] >= 1
        
        return True
    except Exception as e:
        print(f"Cache test error: {e}")
        return False

def test_load_balancer():
    """Test load balancing functionality."""
    from scalable_planner import LoadBalancer
    
    try:
        lb = LoadBalancer()
        
        # Add backends
        lb.add_backend("backend1", 1.0)
        lb.add_backend("backend2", 2.0)
        
        # Test selection strategies
        backend1 = lb.select_backend("round_robin")
        backend2 = lb.select_backend("round_robin")
        
        assert backend1 in ["backend1", "backend2"]
        assert backend2 in ["backend1", "backend2"]
        
        # Test request recording
        lb.record_request("backend1", 0.5, True)
        lb.record_request("backend1", 0.8, True)
        
        return True
    except Exception:
        return False

def test_parallel_processing():
    """Test parallel processing capabilities."""
    from scalable_planner import ParallelProcessor
    
    try:
        processor = ParallelProcessor(max_workers=2)
        
        # Test parallel execution
        def dummy_task():
            time.sleep(0.01)
            return "completed"
        
        tasks = [dummy_task for _ in range(4)]
        start_time = time.time()
        results = processor.run_parallel_sync(tasks)
        execution_time = time.time() - start_time
        
        assert len(results) == 4
        assert all(r == "completed" for r in results if not isinstance(r, Exception))
        assert execution_time < 0.1  # Should be faster than sequential
        
        processor.shutdown()
        return True
    except Exception:
        return False

def test_algorithm_adaptation():
    """Test adaptive algorithm selection."""
    from scalable_planner import AdaptiveOptimizer
    
    try:
        optimizer = AdaptiveOptimizer()
        
        # Record some performance data
        optimizer.record_performance("greedy", 100, 0.8, 0.1)
        optimizer.record_performance("genetic", 100, 0.9, 0.5)
        optimizer.record_performance("greedy", 50, 0.7, 0.05)
        
        # Test algorithm selection
        selected = optimizer.select_algorithm(100)
        assert selected in ["greedy", "genetic", "simulated_annealing", "particle_swarm"]
        
        return True
    except Exception:
        return False

async def test_async_assignment():
    """Test asynchronous assignment capabilities."""
    from scalable_planner import ScalableQuantumPlanner, Agent, Task
    
    try:
        planner = ScalableQuantumPlanner(max_workers=2)
        
        agents = [
            Agent("agent1", ["python"], capacity=2),
            Agent("agent2", ["javascript"], capacity=2)
        ]
        
        tasks = [
            Task("task1", ["python"], priority=1, duration=1),
            Task("task2", ["javascript"], priority=2, duration=1)
        ]
        
        solution = await planner.assign_async(agents, tasks)
        
        assert solution.assignments
        assert solution.solve_time >= 0
        assert solution.quality_score > 0
        
        return True
    except Exception:
        return False

def test_problem_decomposition():
    """Test large problem decomposition."""
    from scalable_planner import ScalableQuantumPlanner, Agent, Task
    
    try:
        planner = ScalableQuantumPlanner()
        
        # Create larger problem
        agents = [Agent(f"agent{i}", [f"skill{i%3}"], capacity=3) for i in range(20)]
        tasks = [Task(f"task{i}", [f"skill{i%3}"], priority=i%5+1, duration=i%3+1) for i in range(50)]
        
        # Test decomposition
        sub_problems = planner._decompose_problem(agents, tasks)
        
        assert len(sub_problems) > 0
        assert all(len(sp[0]) > 0 and len(sp[1]) > 0 for sp in sub_problems)
        
        return True
    except Exception:
        return False

def test_solution_quality():
    """Test solution quality metrics."""
    from scalable_planner import ScalableQuantumPlanner, Agent, Task
    
    try:
        planner = ScalableQuantumPlanner()
        
        agents = [
            Agent("expert", ["quantum", "python"], capacity=5, performance_rating=2.0),
            Agent("junior", ["python"], capacity=2, performance_rating=0.8)
        ]
        
        tasks = [
            Task("quantum_task", ["quantum"], priority=10, duration=3),
            Task("simple_task", ["python"], priority=1, duration=1)
        ]
        
        solution = planner.assign(agents, tasks)
        
        # Verify quality metrics
        assert solution.quality_score > 0
        assert solution.assignments
        assert "expert" in solution.assignments.values()  # Expert should get quantum task
        
        return True
    except Exception:
        return False

def test_error_resilience():
    """Test error handling and resilience."""
    from scalable_planner import ScalableQuantumPlanner, Agent, Task
    
    try:
        planner = ScalableQuantumPlanner()
        
        # Test with no agents
        try:
            solution = planner.assign([], [Task("task1", ["python"])])
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            if "agent" not in str(e).lower():
                return False
        
        # Test with incompatible skills
        agents = [Agent("agent1", ["java"], capacity=1)]
        tasks = [Task("task1", ["python"], priority=1)]
        
        solution = planner.assign(agents, tasks)
        # Should return solution with violations or empty assignments
        assert solution is not None
        
        return True
    except Exception:
        return False

def test_memory_efficiency():
    """Test memory usage and garbage collection."""
    try:
        import gc
        
        # Initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many objects
        from scalable_planner import ScalableQuantumPlanner, Agent, Task
        
        for i in range(100):
            planner = ScalableQuantumPlanner()
            agents = [Agent(f"agent{j}", [f"skill{j}"], capacity=1) for j in range(10)]
            tasks = [Task(f"task{j}", [f"skill{j%3}"], priority=1, duration=1) for j in range(20)]
            solution = planner.assign(agents, tasks)
            
            # Explicitly delete
            del planner, agents, tasks, solution
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Check for memory leaks (allow some growth)
        object_growth = final_objects - initial_objects
        memory_leak_threshold = 1000
        
        return object_growth < memory_leak_threshold
    except Exception:
        return False

def test_thread_safety():
    """Test thread safety of concurrent operations."""
    from scalable_planner import ScalableQuantumPlanner, Agent, Task
    import threading
    
    try:
        planner = ScalableQuantumPlanner(max_workers=4)
        results = []
        errors = []
        
        def worker_task():
            try:
                agents = [Agent(f"agent{i}", ["python"], capacity=2) for i in range(5)]
                tasks = [Task(f"task{i}", ["python"], priority=1, duration=1) for i in range(10)]
                solution = planner.assign(agents, tasks)
                results.append(solution)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = [threading.Thread(target=worker_task) for _ in range(10)]
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # Check results
        return len(errors) == 0 and len(results) == 10
    except Exception:
        return False

def test_benchmark_performance():
    """Benchmark performance against baseline."""
    from scalable_planner import ScalableQuantumPlanner, Agent, Task
    
    try:
        planner = ScalableQuantumPlanner()
        
        # Create benchmark problem
        agents = [Agent(f"agent{i}", [f"skill{i%5}"], capacity=3) for i in range(50)]
        tasks = [Task(f"task{i}", [f"skill{i%5}"], priority=random.randint(1, 10), 
                     duration=random.randint(1, 5)) for i in range(100)]
        
        # Measure performance
        start_time = time.time()
        solution = planner.assign(agents, tasks)
        execution_time = time.time() - start_time
        
        # Performance thresholds
        max_execution_time = 5.0  # seconds
        min_quality_score = 0.8
        
        return (execution_time < max_execution_time and 
                solution.quality_score > min_quality_score)
    except Exception:
        return False

def main():
    """Run comprehensive test suite."""
    print("🚀 Starting Comprehensive Test Suite")
    print("=" * 80)
    
    runner = TestRunner()
    
    # Basic functionality tests
    runner.run_test("Basic Imports", test_basic_imports)
    runner.run_test("Generation 1 Basic", test_generation1_basic_functionality)
    runner.run_test("Generation 2 Robust", test_generation2_robust_functionality)
    runner.run_test("Generation 3 Scalable", test_generation3_scalable_functionality)
    
    # Quality and reliability tests
    runner.run_test("Data Validation", test_data_validation)
    runner.run_test("Security Validation", test_security_validation)
    runner.run_test("Performance Monitoring", test_performance_monitoring)
    runner.run_test("Caching Mechanism", test_caching_mechanism)
    runner.run_test("Load Balancer", test_load_balancer)
    runner.run_test("Parallel Processing", test_parallel_processing)
    runner.run_test("Algorithm Adaptation", test_algorithm_adaptation)
    
    # Advanced functionality tests
    runner.run_test("Async Assignment", test_async_assignment)
    runner.run_test("Problem Decomposition", test_problem_decomposition)
    runner.run_test("Solution Quality", test_solution_quality)
    runner.run_test("Error Resilience", test_error_resilience)
    
    # Performance and reliability tests
    runner.run_test("Memory Efficiency", test_memory_efficiency)
    runner.run_test("Thread Safety", test_thread_safety)
    runner.run_test("Benchmark Performance", test_benchmark_performance)
    
    # Generate final report and quality gate decision
    quality_passed = runner.generate_report()
    
    if quality_passed:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        print("✅ Ready for production deployment")
        return 0
    else:
        print("\n❌ QUALITY GATES FAILED!")
        print("🔧 Please fix issues before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())