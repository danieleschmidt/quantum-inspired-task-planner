"""Performance benchmarks for the quantum task planner."""

import time
import gc
import statistics
from typing import List, Dict, Any
from unittest.mock import Mock, patch

import pytest
import numpy as np

from quantum_planner import QuantumTaskPlanner
from quantum_planner.models import Agent, Task


class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks."""

    @pytest.fixture
    def problem_sizes(self):
        """Different problem sizes for benchmarking."""
        return [
            (5, 8),    # Small: 5 agents, 8 tasks
            (10, 15),  # Medium: 10 agents, 15 tasks  
            (20, 30),  # Large: 20 agents, 30 tasks
            (50, 75),  # Extra Large: 50 agents, 75 tasks
        ]

    @pytest.fixture
    def skill_sets(self):
        """Different skill set configurations."""
        return {
            "homogeneous": ["python", "ml", "devops"],
            "heterogeneous": [
                ["python", "ml"], ["javascript", "react"], 
                ["go", "devops"], ["rust", "systems"],
                ["java", "enterprise"], ["python", "data"]
            ],
            "specialized": [
                ["quantum", "research"], ["ai", "ml", "python"],
                ["blockchain", "solidity"], ["embedded", "c", "cpp"]
            ]
        }

    def generate_test_problem(self, num_agents: int, num_tasks: int, skill_config: str) -> tuple:
        """Generate a test problem with specified parameters."""
        if skill_config == "homogeneous":
            # All agents have same skills
            base_skills = ["python", "ml", "devops"]
            agents = [
                Agent(f"agent_{i}", skills=base_skills, capacity=3)
                for i in range(num_agents)
            ]
            tasks = [
                Task(f"task_{i}", required_skills=["python"], priority=i % 10 + 1, duration=i % 5 + 1)
                for i in range(num_tasks)
            ]
        elif skill_config == "heterogeneous":
            # Diverse skill sets
            skill_options = [
                ["python", "ml"], ["javascript", "react"], ["go", "devops"],
                ["rust", "systems"], ["java", "enterprise"], ["python", "data"]
            ]
            agents = [
                Agent(
                    f"agent_{i}", 
                    skills=skill_options[i % len(skill_options)],
                    capacity=2 + i % 3
                )
                for i in range(num_agents)
            ]
            task_skills = [["python"], ["javascript"], ["go"], ["rust"], ["java"]]
            tasks = [
                Task(
                    f"task_{i}",
                    required_skills=task_skills[i % len(task_skills)],
                    priority=i % 10 + 1,
                    duration=i % 5 + 1
                )
                for i in range(num_tasks)
            ]
        else:  # specialized
            # Highly specialized skills
            specialized_skills = [
                ["quantum", "research"], ["ai", "ml"], ["blockchain"],
                ["embedded", "c"], ["security", "crypto"]
            ]
            agents = [
                Agent(
                    f"agent_{i}",
                    skills=specialized_skills[i % len(specialized_skills)],
                    capacity=2
                )
                for i in range(num_agents)
            ]
            task_skills = [["quantum"], ["ai"], ["blockchain"], ["embedded"], ["security"]]
            tasks = [
                Task(
                    f"task_{i}",
                    required_skills=task_skills[i % len(task_skills)],
                    priority=i % 10 + 1,
                    duration=i % 3 + 1
                )
                for i in range(num_tasks)
            ]
        
        return agents, tasks

    @pytest.mark.benchmark
    def test_assignment_time_scaling(self, problem_sizes, benchmark):
        """Test how assignment time scales with problem size."""
        results = {}
        
        for num_agents, num_tasks in problem_sizes:
            agents, tasks = self.generate_test_problem(num_agents, num_tasks, "homogeneous")
            
            with patch('quantum_planner.backends.get_backend') as mock_get_backend:
                mock_backend = Mock()
                # Generate a reasonable mock solution
                num_vars = num_agents * num_tasks
                mock_solution = {i: 1 if i % (num_agents + 1) == 0 else 0 for i in range(num_vars)}
                mock_backend.solve_qubo.return_value = mock_solution
                mock_get_backend.return_value = mock_backend
                
                planner = QuantumTaskPlanner(backend="mock")
                
                def assignment_task():
                    return planner.assign(
                        agents=agents,
                        tasks=tasks,
                        objective="minimize_makespan"
                    )
                
                result = benchmark.pedantic(
                    assignment_task,
                    iterations=5,
                    rounds=3
                )
                
                results[(num_agents, num_tasks)] = benchmark.stats
        
        # Verify scaling characteristics
        times = [stats.mean for stats in results.values()]
        sizes = [(a * t) for a, t in problem_sizes]
        
        # Check that scaling is reasonable (should be polynomial, not exponential)
        for i in range(1, len(times)):
            scaling_factor = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Scaling should be better than O(n^3)
            assert scaling_factor < size_ratio ** 2.5

    @pytest.mark.benchmark
    def test_memory_scaling(self, problem_sizes):
        """Test memory usage scaling with problem size."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_results = {}
        
        for num_agents, num_tasks in problem_sizes:
            agents, tasks = self.generate_test_problem(num_agents, num_tasks, "homogeneous")
            
            # Force garbage collection before measurement
            gc.collect()
            initial_memory = process.memory_info().rss
            
            with patch('quantum_planner.backends.get_backend') as mock_get_backend:
                mock_backend = Mock()
                num_vars = num_agents * num_tasks
                mock_solution = {i: 1 if i % (num_agents + 1) == 0 else 0 for i in range(num_vars)}
                mock_backend.solve_qubo.return_value = mock_solution
                mock_get_backend.return_value = mock_backend
                
                planner = QuantumTaskPlanner(backend="mock")
                
                solution = planner.assign(
                    agents=agents,
                    tasks=tasks,
                    objective="minimize_makespan"
                )
                
                peak_memory = process.memory_info().rss
                memory_used = peak_memory - initial_memory
                memory_results[(num_agents, num_tasks)] = memory_used / (1024 * 1024)  # MB
        
        # Check memory scaling is reasonable
        for (agents, tasks), memory_mb in memory_results.items():
            problem_size = agents * tasks
            # Memory should be roughly O(n^2) or better
            memory_per_var = memory_mb / problem_size
            assert memory_per_var < 1.0  # Less than 1MB per variable

    @pytest.mark.benchmark
    def test_qubo_construction_performance(self, problem_sizes, benchmark):
        """Benchmark QUBO matrix construction time."""
        from quantum_planner.formulation import QUBOBuilder
        
        for num_agents, num_tasks in problem_sizes:
            agents, tasks = self.generate_test_problem(num_agents, num_tasks, "homogeneous")
            
            def qubo_construction():
                builder = QUBOBuilder()
                builder.add_agents(agents)
                builder.add_tasks(tasks)
                builder.add_objective("minimize_makespan")
                builder.add_constraint("skill_matching")
                builder.add_constraint("capacity_limits")
                return builder.build()
            
            result = benchmark.pedantic(
                qubo_construction,
                iterations=3,
                rounds=2
            )
            
            # QUBO construction should be fast
            assert benchmark.stats.mean < 1.0  # Less than 1 second

    @pytest.mark.benchmark
    def test_backend_comparison(self, benchmark):
        """Compare performance across different backends."""
        num_agents, num_tasks = 15, 20
        agents, tasks = self.generate_test_problem(num_agents, num_tasks, "homogeneous")
        
        backends = ["simulated_annealing", "genetic_algorithm", "tabu_search"]
        results = {}
        
        for backend_type in backends:
            with patch('quantum_planner.backends.get_backend') as mock_get_backend:
                mock_backend = Mock()
                num_vars = num_agents * num_tasks
                mock_solution = {i: 1 if i % (num_agents + 1) == 0 else 0 for i in range(num_vars)}
                
                # Simulate different backend speeds
                if backend_type == "simulated_annealing":
                    time.sleep(0.01)  # Fast
                elif backend_type == "genetic_algorithm":
                    time.sleep(0.05)  # Medium
                else:  # tabu_search
                    time.sleep(0.02)  # Fast-medium
                
                mock_backend.solve_qubo.return_value = mock_solution
                mock_get_backend.return_value = mock_backend
                
                planner = QuantumTaskPlanner(backend=backend_type)
                
                def assignment_task():
                    return planner.assign(
                        agents=agents,
                        tasks=tasks,
                        objective="minimize_makespan"
                    )
                
                result = benchmark.pedantic(
                    assignment_task,
                    iterations=3,
                    rounds=2
                )
                
                results[backend_type] = benchmark.stats.mean
        
        # All backends should complete in reasonable time
        for backend, time_taken in results.items():
            assert time_taken < 1.0  # Less than 1 second

    @pytest.mark.benchmark
    def test_concurrent_assignments(self, benchmark):
        """Test performance under concurrent load."""
        import threading
        import concurrent.futures
        
        num_agents, num_tasks = 10, 15
        agents, tasks = self.generate_test_problem(num_agents, num_tasks, "homogeneous")
        
        def single_assignment():
            with patch('quantum_planner.backends.get_backend') as mock_get_backend:
                mock_backend = Mock()
                num_vars = num_agents * num_tasks
                mock_solution = {i: 1 if i % (num_agents + 1) == 0 else 0 for i in range(num_vars)}
                mock_backend.solve_qubo.return_value = mock_solution
                mock_get_backend.return_value = mock_backend
                
                planner = QuantumTaskPlanner(backend="mock")
                return planner.assign(
                    agents=agents,
                    tasks=tasks,
                    objective="minimize_makespan"
                )
        
        def concurrent_assignments():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(single_assignment) for _ in range(10)]
                results = [future.result() for future in futures]
                return results
        
        results = benchmark(concurrent_assignments)
        assert len(results) == 10
        assert all(result is not None for result in results)

    @pytest.mark.slow
    def test_stress_test(self):
        """Stress test with large problems and extended runtime."""
        # Very large problem
        num_agents, num_tasks = 100, 150
        agents, tasks = self.generate_test_problem(num_agents, num_tasks, "heterogeneous")
        
        with patch('quantum_planner.backends.get_backend') as mock_get_backend:
            mock_backend = Mock()
            num_vars = num_agents * num_tasks
            mock_solution = {i: 1 if i % (num_agents + 1) == 0 else 0 for i in range(num_vars)}
            mock_backend.solve_qubo.return_value = mock_solution
            mock_get_backend.return_value = mock_backend
            
            planner = QuantumTaskPlanner(backend="mock")
            
            start_time = time.time()
            
            solution = planner.assign(
                agents=agents,
                tasks=tasks,
                objective="minimize_makespan",
                constraints={"skill_match": True, "capacity_limit": True}
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time even for large problems
            assert execution_time < 10.0  # Less than 10 seconds
            assert solution is not None
            assert len(solution.assignments) > 0

    def test_solution_quality_vs_time_tradeoff(self):
        """Test the tradeoff between solution quality and computation time."""
        num_agents, num_tasks = 20, 30
        agents, tasks = self.generate_test_problem(num_agents, num_tasks, "homogeneous")
        
        time_limits = [0.1, 0.5, 1.0, 2.0]  # Different time budgets
        results = {}
        
        for time_limit in time_limits:
            with patch('quantum_planner.backends.get_backend') as mock_get_backend:
                mock_backend = Mock()
                
                # Simulate better solutions with longer time
                quality_factor = min(time_limit / 2.0, 1.0)
                num_vars = num_agents * num_tasks
                
                # Better assignment pattern for longer time
                if quality_factor > 0.5:
                    mock_solution = {i: 1 if i % (num_agents + 2) == 0 else 0 for i in range(num_vars)}
                else:
                    mock_solution = {i: 1 if i % (num_agents + 1) == 0 else 0 for i in range(num_vars)}
                
                mock_backend.solve_qubo.return_value = mock_solution
                mock_get_backend.return_value = mock_backend
                
                planner = QuantumTaskPlanner(backend="mock", timeout=time_limit)
                
                start_time = time.time()
                solution = planner.assign(
                    agents=agents,
                    tasks=tasks,
                    objective="minimize_makespan"
                )
                actual_time = time.time() - start_time
                
                results[time_limit] = {
                    "solution": solution,
                    "actual_time": actual_time,
                    "assignments": len(solution.assignments) if solution else 0
                }
        
        # Verify that longer time budgets don't hurt performance
        for time_limit, result in results.items():
            assert result["actual_time"] <= time_limit * 1.5  # Allow some overhead
            assert result["assignments"] > 0


class TestMemoryEfficiency:
    """Test memory efficiency and resource usage."""

    def test_large_matrix_handling(self):
        """Test handling of large QUBO matrices efficiently."""
        # Create a problem that would generate a large matrix
        num_agents = 30
        num_tasks = 45
        
        agents = [Agent(f"agent_{i}", skills=["python"], capacity=3) for i in range(num_agents)]
        tasks = [Task(f"task_{i}", required_skills=["python"], priority=1, duration=1) for i in range(num_tasks)]
        
        from quantum_planner.formulation import QUBOBuilder
        
        builder = QUBOBuilder()
        builder.add_agents(agents)
        builder.add_tasks(tasks)
        
        # Should handle large matrices without excessive memory
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        qubo_matrix = builder.build()
        
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = peak_memory - initial_memory
        
        # Memory usage should be reasonable for matrix size
        matrix_size_mb = (qubo_matrix.shape[0] ** 2 * 8) / (1024 * 1024)  # 8 bytes per float64
        assert memory_used < matrix_size_mb * 3  # Allow 3x overhead for operations

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after operations."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Initial memory
        gc.collect()
        initial_memory = process.memory_info().rss
        
        # Perform multiple assignments
        for i in range(5):
            agents = [Agent(f"agent_{j}", skills=["python"], capacity=2) for j in range(10)]
            tasks = [Task(f"task_{j}", required_skills=["python"], priority=1, duration=1) for j in range(15)]
            
            with patch('quantum_planner.backends.get_backend') as mock_get_backend:
                mock_backend = Mock()
                mock_backend.solve_qubo.return_value = {i: i % 2 for i in range(150)}
                mock_get_backend.return_value = mock_backend
                
                planner = QuantumTaskPlanner(backend="mock")
                solution = planner.assign(agents=agents, tasks=tasks, objective="minimize_makespan")
                
                # Clear references
                del planner, solution, agents, tasks
        
        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        
        # Memory should not have grown excessively
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        assert memory_growth < 50  # Less than 50MB growth