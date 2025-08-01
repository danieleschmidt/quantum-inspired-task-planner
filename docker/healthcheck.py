#!/usr/bin/env python3
"""Health check script for Docker container."""

import sys
import json
import traceback
from pathlib import Path

def check_imports():
    """Check if essential imports work."""
    try:
        import quantum_planner
        return True, f"quantum_planner {quantum_planner.__version__}"
    except ImportError as e:
        return False, f"Import error: {e}"

def check_basic_functionality():
    """Check if basic functionality works."""
    try:
        from quantum_planner import QuantumTaskPlanner, Agent, Task
        
        # Basic instantiation test
        planner = QuantumTaskPlanner(backend="simulator")
        
        # Simple problem test
        agents = [Agent("test_agent", ["python"], 1)]
        tasks = [Task("test_task", ["python"], 5, 1)]
        
        solution = planner.assign(agents, tasks)
        
        if solution and solution.assignments:
            return True, "Basic functionality working"
        else:
            return False, "No solution returned"
            
    except Exception as e:
        return False, f"Functionality error: {e}"

def check_configuration():
    """Check if configuration is valid."""
    try:
        config_path = Path("/app/config")
        if config_path.exists():
            config_files = list(config_path.glob("*.yaml"))
            return True, f"Found {len(config_files)} config files"
        else:
            return False, "Config directory not found"
    except Exception as e:
        return False, f"Config error: {e}"

def main():
    """Run health checks."""
    checks = [
        ("imports", check_imports),
        ("functionality", check_basic_functionality),
        ("configuration", check_configuration)
    ]
    
    results = {}
    all_healthy = True
    
    for name, check_func in checks:
        try:
            healthy, message = check_func()
            results[name] = {
                "healthy": healthy,
                "message": message
            }
            if not healthy:
                all_healthy = False
        except Exception as e:
            results[name] = {
                "healthy": False,
                "message": f"Check failed: {e}",
                "traceback": traceback.format_exc()
            }
            all_healthy = False
    
    # Print results
    if all_healthy:
        print("✓ All health checks passed")
        for name, result in results.items():
            print(f"  {name}: {result['message']}")
        sys.exit(0)
    else:
        print("✗ Some health checks failed")
        for name, result in results.items():
            status = "✓" if result["healthy"] else "✗"
            print(f"  {status} {name}: {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()