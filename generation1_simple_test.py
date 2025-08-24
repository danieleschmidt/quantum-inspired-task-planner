#!/usr/bin/env python3
"""
TERRAGON AUTONOMOUS SDLC - GENERATION 1: SIMPLE IMPLEMENTATION TEST
Tests basic quantum task planner functionality to ensure core features work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from quantum_planner.models import Agent, Task, Solution
from quantum_planner.planner import QuantumTaskPlanner, PlannerConfig


def test_generation1_basic_functionality():
    """Test that Generation 1 basic functionality works."""
    print("🔧 GENERATION 1 TEST: Basic Functionality")
    
    # Create simple agents and tasks
    agents = [
        Agent(agent_id="agent1", skills=["python"], capacity=2),
        Agent(agent_id="agent2", skills=["javascript"], capacity=1),
        Agent(agent_id="agent3", skills=["python", "ml"], capacity=3)
    ]
    
    tasks = [
        Task(task_id="task1", required_skills=["python"], duration=2, priority=5),
        Task(task_id="task2", required_skills=["javascript"], duration=1, priority=3),
        Task(task_id="task3", required_skills=["ml"], duration=3, priority=8)
    ]
    
    # Create planner with basic config
    config = PlannerConfig(backend="simulated_annealing")
    planner = QuantumTaskPlanner(config=config)
    
    print(f"✓ Created {len(agents)} agents and {len(tasks)} tasks")
    print(f"✓ Initialized planner with backend: {config.backend}")
    
    # Test assignment
    try:
        solution = planner.assign(agents, tasks)
        print(f"✓ Task assignment completed successfully")
        print(f"  - Makespan: {solution.makespan}")
        print(f"  - Assignments: {len(solution.assignments)}")
        print(f"  - Backend used: {solution.metadata.get('backend', 'unknown')}")
        return True
    except Exception as e:
        print(f"✗ Assignment failed: {e}")
        return False


def test_generation1_error_handling():
    """Test basic error handling."""
    print("\n🛡️  GENERATION 1 TEST: Error Handling")
    
    try:
        # Test with empty agents
        config = PlannerConfig(backend="simulated_annealing")
        planner = QuantumTaskPlanner(config=config)
        
        agents = []
        tasks = [Task(task_id="task1", required_skills=["python"], duration=1)]
        
        solution = planner.assign(agents, tasks)
        print("✗ Should have raised an error for empty agents")
        return False
    except (ValueError, RuntimeError) as e:
        print(f"✓ Correctly handled empty agents error: {type(e).__name__}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False


def test_generation1_model_validation():
    """Test basic model validation."""
    print("\n📋 GENERATION 1 TEST: Model Validation")
    
    try:
        # Test agent creation
        agent = Agent(agent_id="test", skills=["python"], capacity=1)
        assert agent.agent_id == "test"
        assert "python" in agent.skills
        assert agent.capacity == 1
        print("✓ Agent model validation passed")
        
        # Test task creation
        task = Task(task_id="test", required_skills=["python"], duration=1)
        assert task.task_id == "test"
        assert "python" in task.required_skills
        assert task.duration == 1
        print("✓ Task model validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False


def run_generation1_tests():
    """Run all Generation 1 tests."""
    print("🚀 STARTING GENERATION 1 AUTONOMOUS TESTING")
    print("=" * 50)
    
    tests = [
        test_generation1_basic_functionality,
        test_generation1_error_handling,
        test_generation1_model_validation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🎯 GENERATION 1 TEST SUMMARY")
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ GENERATION 1: ALL TESTS PASSED - READY FOR GENERATION 2")
    else:
        print("❌ GENERATION 1: SOME TESTS FAILED - NEEDS FIXES")
    
    return passed == total


if __name__ == "__main__":
    success = run_generation1_tests()
    sys.exit(0 if success else 1)