#!/usr/bin/env python3
"""Memory profiling script for quantum task planner."""

import gc
import tracemalloc
from typing import List

import numpy as np


@profile  # type: ignore # This decorator is added by memory_profiler
def test_qubo_matrix_creation():
    """Profile memory usage during QUBO matrix creation."""
    # Simulate QUBO matrix creation for different sizes
    matrix_sizes = [10, 25, 50, 100]
    
    for size in matrix_sizes:
        print(f"Creating QUBO matrix of size {size}x{size}")
        
        # Create dense QUBO matrix
        Q = np.random.rand(size, size)
        Q = (Q + Q.T) / 2  # Make symmetric
        
        # Simulate constraint addition
        for i in range(size):
            for j in range(size):
                if i != j:
                    Q[i, j] += np.random.rand() * 0.1
        
        # Force garbage collection
        del Q
        gc.collect()


@profile  # type: ignore
def test_agent_task_scaling():
    """Profile memory usage with increasing agent/task counts."""
    agent_counts = [10, 50, 100, 200]
    
    for num_agents in agent_counts:
        num_tasks = int(num_agents * 1.5)
        print(f"Simulating {num_agents} agents, {num_tasks} tasks")
        
        # Simulate agent data structures
        agents = []
        for i in range(num_agents):
            agent = {
                'id': f'agent_{i}',
                'skills': np.random.choice(['python', 'js', 'ml', 'devops'], 
                                         size=np.random.randint(1, 4)),
                'capacity': np.random.randint(1, 5),
                'efficiency': np.random.rand()
            }
            agents.append(agent)
        
        # Simulate task data structures
        tasks = []
        for i in range(num_tasks):
            task = {
                'id': f'task_{i}',
                'required_skills': np.random.choice(['python', 'js', 'ml', 'devops'], 
                                                   size=np.random.randint(1, 3)),
                'priority': np.random.randint(1, 10),
                'duration': np.random.randint(1, 8),
                'deadline': np.random.randint(8, 24)
            }
            tasks.append(task)
        
        # Simulate assignment matrix
        assignment_matrix = np.zeros((num_agents, num_tasks), dtype=bool)
        
        # Cleanup
        del agents, tasks, assignment_matrix
        gc.collect()


@profile  # type: ignore
def test_solution_storage():
    """Profile memory usage for solution storage and processing."""
    num_solutions = 1000
    solution_size = 50
    
    print(f"Storing {num_solutions} solutions of size {solution_size}")
    
    solutions = []
    for i in range(num_solutions):
        solution = {
            'assignments': np.random.choice([0, 1], size=solution_size).astype(bool),
            'cost': np.random.rand(),
            'makespan': np.random.randint(8, 24),
            'metadata': {
                'solver': 'quantum',
                'solve_time': np.random.rand() * 10,
                'num_reads': np.random.randint(100, 1000)
            }
        }
        solutions.append(solution)
    
    # Simulate solution processing
    best_solutions = sorted(solutions, key=lambda x: x['cost'])[:10]
    
    # Cleanup
    del solutions, best_solutions
    gc.collect()


def main():
    """Run memory profiling tests."""
    print("Starting memory profiling...")
    
    # Start tracing
    tracemalloc.start()
    
    test_qubo_matrix_creation()
    test_agent_task_scaling()
    test_solution_storage()
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()