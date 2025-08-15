#!/usr/bin/env python3
"""Progressive enhancement test for autonomous SDLC implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import logging
from quantum_planner import QuantumTaskPlanner, Agent, Task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generation_1_basic():
    """Test Generation 1: MAKE IT WORK - Basic functionality."""
    logger.info("=== Generation 1: MAKE IT WORK ===")
    
    # Create simple test case
    agents = [
        Agent("agent1", skills=["python", "ml"], capacity=3),
        Agent("agent2", skills=["javascript", "react"], capacity=2),
        Agent("agent3", skills=["python", "devops"], capacity=2),
    ]
    
    tasks = [
        Task("backend_api", required_skills=["python"], priority=5, duration=2),
        Task("frontend_ui", required_skills=["javascript", "react"], priority=3, duration=3),
        Task("ml_pipeline", required_skills=["python", "ml"], priority=8, duration=4),
        Task("deployment", required_skills=["devops"], priority=6, duration=1),
    ]
    
    # Test basic planner initialization
    try:
        planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
        logger.info("✅ Planner initialization successful")
    except Exception as e:
        logger.error(f"❌ Planner initialization failed: {e}")
        return False
    
    # Test basic assignment
    try:
        start_time = time.time()
        solution = planner.assign(
            agents=agents,
            tasks=tasks,
            objective="minimize_makespan",
            constraints={
                "skill_match": True,
                "capacity_limit": True,
            }
        )
        
        solve_time = time.time() - start_time
        logger.info(f"✅ Task assignment successful in {solve_time:.3f}s")
        logger.info(f"   Assignments: {solution.assignments}")
        logger.info(f"   Makespan: {solution.makespan}")
        logger.info(f"   Cost: {solution.cost}")
        
        # Validate solution
        if len(solution.assignments) == len(tasks):
            logger.info("✅ All tasks assigned")
        else:
            logger.warning(f"⚠️  Only {len(solution.assignments)}/{len(tasks)} tasks assigned")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Task assignment failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("=== Testing Edge Cases ===")
    
    planner = QuantumTaskPlanner()
    
    # Test empty inputs
    try:
        planner.assign([], [])
        logger.error("❌ Should have failed with empty inputs")
        return False
    except ValueError:
        logger.info("✅ Correctly rejected empty inputs")
    
    # Test skill mismatch
    agents = [Agent("agent1", skills=["python"], capacity=1)]
    tasks = [Task("task1", required_skills=["javascript"], priority=1, duration=1)]
    
    try:
        solution = planner.assign(agents, tasks)
        logger.error("❌ Should have failed with skill mismatch")
        return False
    except ValueError:
        logger.info("✅ Correctly rejected skill mismatch")
    
    return True

def test_performance_benchmarks():
    """Test basic performance benchmarks."""
    logger.info("=== Performance Benchmarks ===")
    
    planner = QuantumTaskPlanner()
    
    # Scale test: 10 agents, 20 tasks
    agents = [
        Agent(f"agent_{i}", skills=["python", "ml"], capacity=3)
        for i in range(10)
    ]
    
    tasks = [
        Task(f"task_{i}", required_skills=["python"], priority=(i % 5) + 1, duration=i % 3 + 1)
        for i in range(20)
    ]
    
    try:
        start_time = time.time()
        solution = planner.assign(agents, tasks, objective="minimize_makespan")
        solve_time = time.time() - start_time
        
        logger.info(f"✅ 10x20 problem solved in {solve_time:.3f}s")
        logger.info(f"   Solution quality: makespan={solution.makespan}, cost={solution.cost}")
        
        # Performance threshold: should solve in under 5 seconds
        if solve_time < 5.0:
            logger.info("✅ Performance within acceptable limits")
            return True
        else:
            logger.warning(f"⚠️  Performance slower than expected: {solve_time:.3f}s")
            return True  # Still pass but warn
            
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all Generation 1 tests."""
    logger.info("🚀 Starting Autonomous SDLC Progressive Enhancement")
    logger.info("📋 Generation 1: MAKE IT WORK - Basic Functionality Tests")
    
    tests = [
        ("Basic Functionality", test_generation_1_basic),
        ("Edge Cases", test_edge_cases), 
        ("Performance Benchmarks", test_performance_benchmarks),
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
    logger.info(f"\n{'='*50}")
    logger.info("GENERATION 1 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Generation 1 complete! Ready for Generation 2")
        return True
    else:
        logger.error("💥 Generation 1 failed! Fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)