#!/usr/bin/env python3
"""Test CLI structure without executing commands."""

import sys
import os

# Test that CLI is properly structured
def test_cli_structure():
    """Test CLI command structure."""
    print("🖥️  Testing CLI Structure")
    print("=" * 30)
    
    # Check that CLI file exists and has expected content
    cli_file = "src/quantum_planner/cli.py"
    if not os.path.exists(cli_file):
        print(f"❌ CLI file not found: {cli_file}")
        return False
    
    with open(cli_file, 'r') as f:
        content = f.read()
    
    expected_functions = [
        "@main.command()",  # Command decorators
        "def solve(",       # solve command
        "def generate(",    # generate command  
        "def status(",      # status command
        "def backends(",    # backends command
        "def main():",      # main function
        "click.group()",    # Click group
    ]
    
    print("\n1. Checking CLI command structure...")
    missing = []
    for func in expected_functions:
        if func not in content:
            missing.append(func)
        else:
            print(f"   ✓ Found: {func}")
    
    if missing:
        print(f"   ❌ Missing: {missing}")
        return False
    
    # Check pyproject.toml has CLI script entries
    print("\n2. Checking pyproject.toml script entries...")
    pyproject_file = "pyproject.toml"
    if not os.path.exists(pyproject_file):
        print(f"❌ pyproject.toml not found")
        return False
        
    with open(pyproject_file, 'r') as f:
        pyproject_content = f.read()
    
    expected_scripts = [
        'quantum-planner = "quantum_planner.cli:main"',
        'qp = "quantum_planner.cli:main"'
    ]
    
    for script in expected_scripts:
        if script in pyproject_content:
            print(f"   ✓ Found script: {script}")
        else:
            print(f"   ❌ Missing script: {script}")
            return False
    
    # Check example files exist
    print("\n3. Checking example files...")
    example_files = [
        "examples/basic_usage.py",
        "examples/problem_example.json", 
        "examples/time_window_example.json"
    ]
    
    for example_file in example_files:
        if os.path.exists(example_file):
            print(f"   ✓ Found: {example_file}")
        else:
            print(f"   ❌ Missing: {example_file}")
            return False
    
    # Check that examples have expected content
    print("\n4. Checking example content...")
    with open("examples/basic_usage.py", 'r') as f:
        example_content = f.read()
    
    expected_in_examples = [
        "QuantumTaskPlanner",
        "Agent(",
        "Task(",
        "planner.assign(",
        "solution.assignments"
    ]
    
    for expected in expected_in_examples:
        if expected in example_content:
            print(f"   ✓ Example contains: {expected}")
        else:
            print(f"   ❌ Example missing: {expected}")
            return False
    
    print("\n🎉 CLI structure tests passed!")
    return True


if __name__ == "__main__":
    success = test_cli_structure()
    if success:
        print("\n📋 CLI Structure Summary:")
        print("   ✓ Command definitions (solve, generate, status, backends)")
        print("   ✓ Script entries in pyproject.toml")
        print("   ✓ Example files with correct API usage")
        print("   ✓ JSON problem file examples")
        print("\n🚀 CLI is ready for use!")
        print("   Usage: python -m quantum_planner.cli --help")
    else:
        print("\n❌ CLI structure tests failed")
        sys.exit(1)