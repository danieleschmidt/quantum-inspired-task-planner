#!/usr/bin/env python3
"""Generation 2: MAKE IT ROBUST - Comprehensive error handling and validation testing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import logging
from quantum_planner import QuantumTaskPlanner, Agent, Task
from quantum_planner.validation import InputValidator, format_validation_report
from quantum_planner.security import security_manager, SecurityLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_comprehensive_validation():
    """Test comprehensive input validation."""
    logger.info("=== Generation 2: Comprehensive Validation ===")
    
    validator = InputValidator(strict_mode=False)
    
    # Test valid inputs
    agents = [
        Agent("agent1", skills=["python", "ml"], capacity=3),
        Agent("agent2", skills=["javascript"], capacity=2),
    ]
    
    tasks = [
        Task("task1", required_skills=["python"], priority=5, duration=2),
        Task("task2", required_skills=["javascript"], priority=3, duration=1),
    ]
    
    try:
        # Test individual validations
        agent_report = validator.validate_agents(agents)
        task_report = validator.validate_tasks(tasks)
        compat_report = validator.validate_compatibility(agents, tasks)
        
        if agent_report.is_valid and task_report.is_valid and compat_report.is_valid:
            logger.info("✅ Valid inputs passed validation")
        else:
            logger.error("❌ Valid inputs failed validation")
            return False
            
        # Test comprehensive validation
        overall_report = validator.validate_all(agents, tasks)
        if overall_report.is_valid:
            logger.info("✅ Comprehensive validation passed")
        else:
            logger.error(f"❌ Comprehensive validation failed:")
            logger.error(format_validation_report(overall_report))
            return False
            
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")
        return False
    
    return True


def test_error_handling():
    """Test robust error handling."""
    logger.info("=== Generation 2: Error Handling ===")
    
    planner = QuantumTaskPlanner()
    validator = InputValidator(strict_mode=False)
    
    # Test case 1: Empty agent ID
    try:
        invalid_agents = [Agent("", skills=["python"], capacity=1)]
        tasks = [Task("task1", required_skills=["python"], priority=1, duration=1)]
        
        report = validator.validate_agents(invalid_agents)
        if not report.is_valid:
            logger.info("✅ Correctly detected empty agent ID")
        else:
            logger.error("❌ Failed to detect empty agent ID")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False
    
    # Test case 2: Invalid task priority (test validation separately)
    try:
        agents = [Agent("agent1", skills=["python"], capacity=1)]
        valid_task = Task("task1", required_skills=["python"], priority=1, duration=1)
        
        # Manually set invalid priority to test validation
        object.__setattr__(valid_task, 'priority', -1)
        invalid_tasks = [valid_task]
        
        report = validator.validate_tasks(invalid_tasks)
        if not report.is_valid:
            logger.info("✅ Correctly detected invalid priority")
        else:
            logger.error("❌ Failed to detect invalid priority")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False
    
    # Test case 3: Skill mismatch
    try:
        agents = [Agent("agent1", skills=["python"], capacity=1)]
        tasks = [Task("task1", required_skills=["javascript"], priority=1, duration=1)]
        
        report = validator.validate_compatibility(agents, tasks)
        if not report.is_valid:
            logger.info("✅ Correctly detected skill mismatch")
        else:
            logger.error("❌ Failed to detect skill mismatch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False
    
    return True


def test_security_features():
    """Test security features."""
    logger.info("=== Generation 2: Security Features ===")
    
    try:
        # Test session management
        user_id = "test_user"
        token = security_manager.generate_session_token(user_id)
        
        if len(token) >= 32:  # Should be a secure token
            logger.info("✅ Secure session token generated")
        else:
            logger.error("❌ Insecure session token")
            return False
        
        # Test token validation
        session = security_manager.validate_session_token(token)
        if session and session["user_id"] == user_id:
            logger.info("✅ Session token validation works")
        else:
            logger.error("❌ Session token validation failed")
            return False
        
        # Test rate limiting
        identifier = "test_ip"
        for i in range(5):
            allowed = security_manager.check_rate_limit(identifier, max_requests=3, window_seconds=60)
            if i < 3 and not allowed:
                logger.error("❌ Rate limiting too restrictive")
                return False
            elif i >= 3 and allowed:
                logger.error("❌ Rate limiting not working")
                return False
        
        logger.info("✅ Rate limiting works correctly")
        
        # Test input sanitization
        dangerous_input = "<script>alert('xss')</script>"
        sanitized = security_manager.sanitize_input(dangerous_input)
        if "<script>" not in sanitized:
            logger.info("✅ Input sanitization works")
        else:
            logger.error("❌ Input sanitization failed")
            return False
        
        # Test problem size validation
        if security_manager.validate_problem_size(100, 200):
            logger.info("✅ Normal problem size accepted")
        else:
            logger.error("❌ Normal problem size rejected")
            return False
        
        if not security_manager.validate_problem_size(10000, 20000):
            logger.info("✅ Excessive problem size rejected")
        else:
            logger.error("❌ Excessive problem size accepted")
            return False
        
    except Exception as e:
        logger.error(f"❌ Security test failed: {e}")
        return False
    
    return True


def test_logging_monitoring():
    """Test logging and monitoring."""
    logger.info("=== Generation 2: Logging and Monitoring ===")
    
    try:
        # Test security event logging
        initial_events = len(security_manager.audit_log)
        
        security_manager.log_security_event(
            event_type="test_event",
            severity=SecurityLevel.LOW,
            user_id="test_user",
            details={"test": "data"}
        )
        
        if len(security_manager.audit_log) > initial_events:
            logger.info("✅ Security event logging works")
        else:
            logger.error("❌ Security event logging failed")
            return False
        
        # Test security metrics
        metrics = security_manager.get_security_metrics()
        required_metrics = [
            "total_events", "events_last_hour", "events_last_day",
            "active_sessions", "blocked_ips", "rate_limited_identifiers"
        ]
        
        for metric in required_metrics:
            if metric not in metrics:
                logger.error(f"❌ Missing security metric: {metric}")
                return False
        
        logger.info("✅ Security metrics collection works")
        
        # Test audit trail
        recent_events = security_manager.get_recent_events(limit=10)
        if recent_events:
            logger.info("✅ Audit trail accessible")
        else:
            logger.error("❌ Audit trail empty")
            return False
        
    except Exception as e:
        logger.error(f"❌ Logging and monitoring test failed: {e}")
        return False
    
    return True


def test_resilience_recovery():
    """Test system resilience and recovery."""
    logger.info("=== Generation 2: Resilience and Recovery ===")
    
    try:
        planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
        
        # Test with challenging input
        agents = [Agent(f"agent_{i}", skills=["python"], capacity=2) for i in range(5)]
        tasks = [Task(f"task_{i}", required_skills=["python"], priority=i+1, duration=1) for i in range(20)]
        
        # This should stress the system but still work
        start_time = time.time()
        solution = planner.assign(agents, tasks, objective="minimize_makespan")
        solve_time = time.time() - start_time
        
        if solution and len(solution.assignments) > 0:
            logger.info(f"✅ System handled challenging input in {solve_time:.3f}s")
        else:
            logger.error("❌ System failed with challenging input")
            return False
        
        # Test health status
        health = planner.get_health_status()
        if health["overall_status"] in ["healthy", "degraded"]:
            logger.info("✅ Health monitoring works")
        else:
            logger.error("❌ System health critical")
            return False
        
        # Test performance stats
        perf_stats = planner.get_performance_stats()
        if "planner" in perf_stats:
            logger.info("✅ Performance monitoring works")
        else:
            logger.error("❌ Performance monitoring failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Resilience test failed: {e}")
        return False
    
    return True


def test_solution_validation():
    """Test solution validation."""
    logger.info("=== Generation 2: Solution Validation ===")
    
    try:
        planner = QuantumTaskPlanner()
        validator = InputValidator()
        
        # Create a valid problem
        agents = [
            Agent("agent1", skills=["python"], capacity=2),
            Agent("agent2", skills=["javascript"], capacity=2),
        ]
        
        tasks = [
            Task("task1", required_skills=["python"], priority=5, duration=1),
            Task("task2", required_skills=["javascript"], priority=3, duration=1),
        ]
        
        # Get solution
        solution = planner.assign(agents, tasks)
        
        # Validate solution
        solution_report = validator.validate_solution(solution, agents, tasks)
        
        if solution_report.is_valid:
            logger.info("✅ Solution validation passed")
        else:
            logger.error("❌ Solution validation failed:")
            logger.error(format_validation_report(solution_report))
            return False
        
        # Test with invalid solution (manually create)
        from quantum_planner.models import Solution
        invalid_solution = Solution(
            assignments={"task1": "nonexistent_agent"},
            makespan=10,
            cost=5,
            backend_used="test"
        )
        
        invalid_report = validator.validate_solution(invalid_solution, agents, tasks)
        if not invalid_report.is_valid:
            logger.info("✅ Invalid solution correctly detected")
        else:
            logger.error("❌ Failed to detect invalid solution")
            return False
        
    except Exception as e:
        logger.error(f"❌ Solution validation test failed: {e}")
        return False
    
    return True


def main():
    """Run all Generation 2 tests."""
    logger.info("🚀 Generation 2: MAKE IT ROBUST")
    logger.info("📋 Comprehensive Error Handling, Validation, Security & Monitoring")
    
    tests = [
        ("Comprehensive Validation", test_comprehensive_validation),
        ("Error Handling", test_error_handling),
        ("Security Features", test_security_features),
        ("Logging & Monitoring", test_logging_monitoring),
        ("Resilience & Recovery", test_resilience_recovery),
        ("Solution Validation", test_solution_validation),
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
    logger.info("GENERATION 2 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Generation 2 complete! System is now ROBUST")
        logger.info("📈 Ready for Generation 3: Performance Optimization")
        return True
    else:
        logger.error("💥 Generation 2 failed! Fix robustness issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)