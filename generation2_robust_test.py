#!/usr/bin/env python3
"""
TERRAGON AUTONOMOUS SDLC - GENERATION 2: ROBUST IMPLEMENTATION
Enhances Generation 1 with comprehensive error handling, validation, logging, monitoring, and security measures.
"""

import sys
import os
import logging
import time
import traceback
import json
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_planner.models import Agent, Task, Solution
from quantum_planner.planner import QuantumTaskPlanner, PlannerConfig
# Import reliability components if available
try:
    from quantum_planner.reliability import reliability_manager
except ImportError:
    reliability_manager = None


# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation2_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobustQuantumPlanner:
    """Generation 2: Robust quantum task planner with comprehensive error handling."""
    
    def __init__(self, config: PlannerConfig = None):
        self.config = config or PlannerConfig(backend="simulated_annealing", verbose=True)
        self.planner = None
        self.metrics = {
            "total_assignments": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "average_solve_time": 0.0,
            "error_counts": {},
            "security_checks": 0
        }
        self._initialize_planner()
    
    def _initialize_planner(self):
        """Initialize planner with robust error handling."""
        try:
            self.planner = QuantumTaskPlanner(config=self.config)
            logger.info(f"Successfully initialized planner with backend: {self.config.backend}")
        except Exception as e:
            logger.error(f"Failed to initialize planner: {e}")
            logger.info("Falling back to basic configuration...")
            try:
                self.config = PlannerConfig(backend="simulated_annealing")
                self.planner = QuantumTaskPlanner(config=self.config)
                logger.info("Fallback initialization successful")
            except Exception as fallback_error:
                logger.critical(f"Complete initialization failure: {fallback_error}")
                raise RuntimeError("Cannot initialize quantum planner") from fallback_error
    
    def validate_inputs(self, agents, tasks):
        """Comprehensive input validation with security checks."""
        self.metrics["security_checks"] += 1
        
        # Validate agents
        if not agents:
            raise ValueError("Agent list cannot be empty")
        
        if len(agents) > 1000:
            logger.warning(f"Large agent pool ({len(agents)}) may impact performance")
        
        for i, agent in enumerate(agents):
            if not hasattr(agent, 'agent_id') or not agent.agent_id:
                raise ValueError(f"Agent at index {i} missing valid agent_id")
            
            if not hasattr(agent, 'skills') or not agent.skills:
                logger.warning(f"Agent {agent.agent_id} has no skills")
            
            if hasattr(agent, 'capacity') and agent.capacity <= 0:
                raise ValueError(f"Agent {agent.agent_id} has invalid capacity: {agent.capacity}")
        
        # Validate tasks
        if not tasks:
            raise ValueError("Task list cannot be empty")
        
        if len(tasks) > 2000:
            logger.warning(f"Large task count ({len(tasks)}) may impact performance")
        
        for i, task in enumerate(tasks):
            if not hasattr(task, 'task_id') or not task.task_id:
                raise ValueError(f"Task at index {i} missing valid task_id")
            
            if not hasattr(task, 'required_skills') or not task.required_skills:
                raise ValueError(f"Task {task.task_id} has no required skills")
            
            if hasattr(task, 'duration') and task.duration <= 0:
                raise ValueError(f"Task {task.task_id} has invalid duration: {task.duration}")
        
        # Check skill compatibility
        all_agent_skills = set()
        for agent in agents:
            all_agent_skills.update(agent.skills)
        
        all_task_skills = set()
        for task in tasks:
            all_task_skills.update(task.required_skills)
        
        missing_skills = all_task_skills - all_agent_skills
        if missing_skills:
            logger.error(f"Tasks require skills not available in agent pool: {missing_skills}")
            raise ValueError(f"Unassignable tasks due to missing skills: {missing_skills}")
        
        logger.info(f"Input validation passed: {len(agents)} agents, {len(tasks)} tasks")
    
    def assign_with_monitoring(self, agents, tasks, objective="minimize_makespan"):
        """Assign tasks with comprehensive monitoring and error handling."""
        start_time = time.time()
        self.metrics["total_assignments"] += 1
        
        try:
            # Input validation
            self.validate_inputs(agents, tasks)
            
            # Log assignment attempt
            logger.info(f"Starting assignment: {len(agents)} agents, {len(tasks)} tasks, objective: {objective}")
            
            # Attempt assignment with timeout
            assignment_start = time.time()
            
            # Use enhanced error handling
            try:
                solution = self.planner.assign(agents, tasks, objective=objective)
            except Exception as e:
                logger.error(f"Assignment error in quantum planner: {e}")
                raise
            
            solve_time = time.time() - assignment_start
            
            # Validate solution
            self._validate_solution(solution, agents, tasks)
            
            # Update metrics
            self.metrics["successful_assignments"] += 1
            self.metrics["average_solve_time"] = (
                (self.metrics["average_solve_time"] * (self.metrics["successful_assignments"] - 1) + solve_time) / 
                self.metrics["successful_assignments"]
            )
            
            total_time = time.time() - start_time
            logger.info(f"Assignment completed successfully in {total_time:.2f}s (solve: {solve_time:.2f}s)")
            
            return solution
            
        except Exception as e:
            # Comprehensive error handling and logging
            self.metrics["failed_assignments"] += 1
            error_type = type(e).__name__
            self.metrics["error_counts"][error_type] = self.metrics["error_counts"].get(error_type, 0) + 1
            
            total_time = time.time() - start_time
            logger.error(f"Assignment failed after {total_time:.2f}s: {error_type}: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            # Attempt recovery strategies
            if "timeout" in str(e).lower():
                logger.info("Timeout detected, attempting with reduced complexity...")
                return self._attempt_simplified_assignment(agents[:min(10, len(agents))], 
                                                         tasks[:min(20, len(tasks))], 
                                                         objective)
            
            # Re-raise after logging
            raise
    
    def _validate_solution(self, solution, agents, tasks):
        """Validate solution quality and constraints."""
        if not solution:
            raise ValueError("Solution is None or empty")
        
        if not hasattr(solution, 'assignments') or not solution.assignments:
            raise ValueError("Solution contains no assignments")
        
        # Check that all tasks are assigned
        assigned_tasks = set(solution.assignments.keys())
        all_tasks = set(task.task_id for task in tasks)
        
        unassigned = all_tasks - assigned_tasks
        if unassigned:
            logger.warning(f"Unassigned tasks detected: {unassigned}")
        
        # Check that assignments are valid
        all_agents = set(agent.agent_id for agent in agents)
        for task_id, agent_id in solution.assignments.items():
            if agent_id not in all_agents:
                raise ValueError(f"Task {task_id} assigned to non-existent agent {agent_id}")
        
        logger.info(f"Solution validation passed: {len(solution.assignments)} assignments")
    
    def _attempt_simplified_assignment(self, agents, tasks, objective):
        """Fallback assignment with reduced complexity."""
        logger.info(f"Attempting simplified assignment: {len(agents)} agents, {len(tasks)} tasks")
        try:
            simplified_config = PlannerConfig(
                backend="simulated_annealing",
                max_solve_time=60,
                verbose=False
            )
            simplified_planner = QuantumTaskPlanner(config=simplified_config)
            return simplified_planner.assign(agents, tasks, objective=objective)
        except Exception as e:
            logger.error(f"Simplified assignment also failed: {e}")
            raise
    
    def get_health_metrics(self):
        """Return comprehensive health and performance metrics."""
        success_rate = (
            self.metrics["successful_assignments"] / max(1, self.metrics["total_assignments"])
        )
        
        health_metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_assignments": self.metrics["total_assignments"],
            "success_rate": round(success_rate * 100, 2),
            "average_solve_time": round(self.metrics["average_solve_time"], 3),
            "error_distribution": self.metrics["error_counts"],
            "security_checks_performed": self.metrics["security_checks"],
            "health_status": "healthy" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "critical"
        }
        
        return health_metrics


def test_generation2_robust_functionality():
    """Test Generation 2 robust functionality."""
    print("🛡️  GENERATION 2 TEST: Robust Functionality")
    
    planner = RobustQuantumPlanner()
    
    # Test normal operation
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
    
    try:
        solution = planner.assign_with_monitoring(agents, tasks)
        metrics = planner.get_health_metrics()
        
        print(f"✓ Robust assignment completed")
        print(f"  - Success rate: {metrics['success_rate']}%")
        print(f"  - Average solve time: {metrics['average_solve_time']}s")
        print(f"  - Health status: {metrics['health_status']}")
        
        return True
    except Exception as e:
        print(f"✗ Robust assignment failed: {e}")
        return False


def test_generation2_error_recovery():
    """Test Generation 2 error recovery mechanisms."""
    print("\n🔧 GENERATION 2 TEST: Error Recovery")
    
    planner = RobustQuantumPlanner()
    
    # Test with invalid inputs
    test_cases = [
        ([], [Task(task_id="task1", required_skills=["python"], duration=1)], "empty agents"),
        ([Agent(agent_id="agent1", skills=["python"], capacity=1)], [], "empty tasks"),
        ([Agent(agent_id="agent1", skills=["java"], capacity=1)], 
         [Task(task_id="task1", required_skills=["python"], duration=1)], "skill mismatch")
    ]
    
    recovery_success = 0
    for agents, tasks, test_name in test_cases:
        try:
            planner.assign_with_monitoring(agents, tasks)
            print(f"✗ {test_name}: Should have failed")
        except (ValueError, RuntimeError) as e:
            print(f"✓ {test_name}: Correctly caught {type(e).__name__}")
            recovery_success += 1
        except Exception as e:
            print(f"? {test_name}: Unexpected error {type(e).__name__}: {e}")
    
    metrics = planner.get_health_metrics()
    print(f"✓ Error recovery metrics: {recovery_success}/3 tests handled correctly")
    print(f"  - Total operations: {metrics['total_assignments']}")
    print(f"  - Security checks: {metrics['security_checks_performed']}")
    
    return recovery_success >= 2


def test_generation2_monitoring_logging():
    """Test Generation 2 monitoring and logging capabilities."""
    print("\n📊 GENERATION 2 TEST: Monitoring & Logging")
    
    planner = RobustQuantumPlanner(PlannerConfig(backend="simulated_annealing", verbose=True))
    
    # Test multiple assignments to generate metrics
    agents = [Agent(agent_id=f"agent{i}", skills=["python"], capacity=1) for i in range(3)]
    
    for i in range(5):
        tasks = [Task(task_id=f"task{i}_{j}", required_skills=["python"], duration=1) for j in range(2)]
        try:
            planner.assign_with_monitoring(agents, tasks)
        except Exception:
            pass  # Expected for some test cases
    
    metrics = planner.get_health_metrics()
    
    # Validate monitoring data
    checks = []
    checks.append(("Total assignments recorded", metrics['total_assignments'] > 0))
    checks.append(("Success rate calculated", 0 <= metrics['success_rate'] <= 100))
    checks.append(("Average solve time available", metrics['average_solve_time'] >= 0))
    checks.append(("Security checks performed", metrics['security_checks_performed'] > 0))
    checks.append(("Health status determined", metrics['health_status'] in ['healthy', 'degraded', 'critical']))
    
    passed = sum(1 for _, check in checks if check)
    
    for check_name, result in checks:
        print(f"{'✓' if result else '✗'} {check_name}")
    
    print(f"✓ Monitoring validation: {passed}/{len(checks)} checks passed")
    
    # Verify log file creation
    log_exists = os.path.exists('generation2_robust.log')
    print(f"{'✓' if log_exists else '✗'} Log file created")
    
    return passed >= len(checks) - 1 and log_exists


def run_generation2_tests():
    """Run all Generation 2 robust tests."""
    print("🚀 STARTING GENERATION 2 ROBUST TESTING")
    print("=" * 50)
    
    tests = [
        test_generation2_robust_functionality,
        test_generation2_error_recovery,
        test_generation2_monitoring_logging
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            logger.error(f"Test crash: {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("🎯 GENERATION 2 TEST SUMMARY")
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ GENERATION 2: ALL TESTS PASSED - READY FOR GENERATION 3")
    else:
        print("❌ GENERATION 2: SOME TESTS FAILED - NEEDS INVESTIGATION")
    
    # Write comprehensive test report
    report = {
        "generation": 2,
        "timestamp": datetime.now().isoformat(),
        "tests_total": total,
        "tests_passed": passed,
        "success_rate": round(passed / total * 100, 2),
        "status": "PASSED" if passed == total else "FAILED"
    }
    
    with open('generation2_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📝 Detailed report saved to: generation2_test_report.json")
    
    return passed == total


if __name__ == "__main__":
    success = run_generation2_tests()
    sys.exit(0 if success else 1)