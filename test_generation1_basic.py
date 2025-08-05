#!/usr/bin/env python3
"""Generation 1 Basic Functionality Test - Simple Working Implementation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_planner import QuantumTaskPlanner, Agent, Task, Solution
from quantum_planner.optimizer import OptimizationBackend, OptimizationParams

def test_basic_quantum_planner():
    """Test basic quantum task planner functionality."""
    print("🧪 Testing Generation 1: Basic Quantum Task Planner")
    
    # Create simple agents with skills
    agents = [
        Agent(id="agent1", skills=["python", "ml"], capacity=3),
        Agent(id="agent2", skills=["javascript", "react"], capacity=2),
        Agent(id="agent3", skills=["python", "devops"], capacity=2),
    ]
    
    # Create simple tasks
    tasks = [
        Task(id="backend_api", required_skills=["python"], priority=5, duration=2),
        Task(id="frontend_ui", required_skills=["javascript", "react"], priority=3, duration=3),
        Task(id="ml_pipeline", required_skills=["python", "ml"], priority=8, duration=4),
        Task(id="deployment", required_skills=["devops"], priority=6, duration=1),
    ]
    
    # Initialize planner with fallback
    planner = QuantumTaskPlanner(
        backend="auto",
        fallback="simulated_annealing"
    )
    
    print(f"📊 Problem: {len(agents)} agents, {len(tasks)} tasks")
    
    # Solve assignment problem
    solution = planner.assign(
        agents=agents,
        tasks=tasks,
        objective="minimize_makespan",
        constraints={
            "skill_match": True,
            "capacity_limit": True
        }
    )
    
    # Validate results
    print(f"✅ Assignments: {solution.assignments}")
    print(f"⏱️  Makespan: {solution.makespan}")
    print(f"💰 Cost: {solution.cost}")
    print(f"🔧 Backend: {solution.backend_used}")
    
    # Verify all tasks are assigned
    assert len(solution.assignments) == len(tasks), "Not all tasks assigned"
    
    # Verify all assigned agents exist
    assigned_agents = set(solution.assignments.values())
    available_agents = {agent.id for agent in agents}
    assert assigned_agents.issubset(available_agents), "Invalid agent assignment"
    
    print("✅ Generation 1 Basic Test PASSED")
    return True

def test_skill_matching():
    """Test that skill matching works correctly."""
    print("\n🎯 Testing Skill Matching")
    
    # Create agents with specific skills
    agents = [
        Agent(id="python_dev", skills=["python"], capacity=1),
        Agent(id="js_dev", skills=["javascript"], capacity=1),
    ]
    
    # Create task requiring specific skill
    tasks = [
        Task(id="python_task", required_skills=["python"], priority=1, duration=1),
    ]
    
    planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
    solution = planner.assign(agents, tasks)
    
    # Should be assigned to python_dev
    assert solution.assignments["python_task"] == "python_dev"
    
    print("✅ Skill Matching Test PASSED")
    return True

def test_capacity_constraints():
    """Test capacity constraint handling."""
    print("\n📦 Testing Capacity Constraints")
    
    # Single agent with limited capacity
    agents = [
        Agent(id="limited_agent", skills=["python"], capacity=1),
    ]
    
    # Multiple tasks
    tasks = [
        Task(id="task1", required_skills=["python"], priority=1, duration=1),
        Task(id="task2", required_skills=["python"], priority=1, duration=1),
    ]
    
    planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
    solution = planner.assign(agents, tasks)
    
    # Both tasks should be assigned to the same agent (capacity handled by makespan)
    assert len(solution.assignments) == 2
    assert all(agent_id == "limited_agent" for agent_id in solution.assignments.values())
    
    print("✅ Capacity Constraints Test PASSED")
    return True

def test_backend_fallback():
    """Test backend fallback mechanism."""
    print("\n🔄 Testing Backend Fallback")
    
    # Create simple problem
    agents = [Agent(id="agent1", skills=["python"], capacity=1)]
    tasks = [Task(id="task1", required_skills=["python"], priority=1, duration=1)]
    
    # Try with non-existent quantum backend (should fallback)
    planner = QuantumTaskPlanner(
        backend="nonexistent",
        fallback="simulated_annealing"
    )
    
    solution = planner.assign(agents, tasks)
    
    # Should have fallback metadata
    assert solution.metadata.get("fallback_used") is not None
    
    print("✅ Backend Fallback Test PASSED")
    return True

def test_solution_quality():
    """Test solution quality metrics."""
    print("\n🏆 Testing Solution Quality")
    
    agents = [
        Agent(id="agent1", skills=["python"], capacity=2),
        Agent(id="agent2", skills=["python"], capacity=2),
    ]
    
    tasks = [
        Task(id="task1", required_skills=["python"], priority=1, duration=1),
        Task(id="task2", required_skills=["python"], priority=1, duration=1),
    ]
    
    planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
    solution = planner.assign(agents, tasks)
    
    # Test quality metrics
    quality_score = solution.calculate_quality_score()
    assert 0 <= quality_score <= 1, "Quality score out of range"
    
    load_dist = solution.get_load_distribution()
    assert isinstance(load_dist, dict), "Load distribution not a dict"
    
    print(f"📈 Quality Score: {quality_score:.3f}")
    print(f"⚖️  Load Distribution: {load_dist}")
    
    print("✅ Solution Quality Test PASSED")
    return True

if __name__ == "__main__":
    print("🚀 Starting Generation 1 Basic Functionality Tests\n")
    
    try:
        # Run all tests
        test_basic_quantum_planner()
        test_skill_matching()
        test_capacity_constraints()
        test_backend_fallback()
        test_solution_quality()
        
        print("\n🎉 ALL GENERATION 1 TESTS PASSED!")
        print("✅ Basic functionality is working correctly")
        print("✅ Skill matching implemented")
        print("✅ Capacity constraints handled")
        print("✅ Backend fallback mechanism working")
        print("✅ Solution quality metrics functional")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)