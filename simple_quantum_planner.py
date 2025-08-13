#!/usr/bin/env python3
"""
Simple Quantum Task Planner - Generation 1: Make It Work
A simplified interface for quantum-inspired task scheduling with minimal setup.
"""

from typing import List, Dict, Optional, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_planner import QuantumTaskPlanner, Agent, Task, Solution


class SimpleQuantumPlanner:
    """
    Ultra-simplified interface for quantum task scheduling.
    Handles all complexity internally with smart defaults.
    """
    
    def __init__(self):
        """Initialize with automatic backend selection."""
        self.planner = QuantumTaskPlanner(backend="auto", fallback="simulated_annealing")
    
    def assign_tasks(
        self, 
        agents: List[Dict[str, Any]], 
        tasks: List[Dict[str, Any]],
        minimize: str = "time"
    ) -> Dict[str, Any]:
        """
        Assign tasks to agents with minimal configuration.
        
        Args:
            agents: List of dicts with 'id', 'skills', 'capacity'
            tasks: List of dicts with 'id', 'skills', 'priority', 'duration'  
            minimize: What to optimize - 'time' or 'cost'
            
        Returns:
            Dict with 'assignments', 'completion_time', 'success'
            
        Example:
            >>> planner = SimpleQuantumPlanner()
            >>> agents = [
            ...     {'id': 'alice', 'skills': ['python'], 'capacity': 2},
            ...     {'id': 'bob', 'skills': ['javascript'], 'capacity': 1}
            ... ]
            >>> tasks = [
            ...     {'id': 'backend', 'skills': ['python'], 'priority': 5, 'duration': 3},
            ...     {'id': 'frontend', 'skills': ['javascript'], 'priority': 3, 'duration': 2}
            ... ]
            >>> result = planner.assign_tasks(agents, tasks)
            >>> print(result['assignments'])
        """
        try:
            # Convert to internal models
            agent_objects = []
            for a in agents:
                agent_objects.append(Agent(
                    agent_id=a['id'],
                    skills=a['skills'],
                    capacity=a.get('capacity', 1)
                ))
            
            task_objects = []
            for t in tasks:
                task_objects.append(Task(
                    task_id=t['id'],
                    required_skills=t['skills'],
                    priority=t.get('priority', 1),
                    duration=t.get('duration', 1)
                ))
            
            # Map minimize parameter to objective
            objective_map = {
                'time': 'minimize_makespan',
                'cost': 'minimize_cost'
            }
            objective = objective_map.get(minimize, 'minimize_makespan')
            
            # Solve with automatic constraints
            solution = self.planner.assign(
                agents=agent_objects,
                tasks=task_objects,
                objective=objective,
                constraints={
                    'skill_match': True,
                    'capacity_limit': True
                }
            )
            
            return {
                'assignments': solution.assignments,
                'completion_time': solution.makespan,
                'total_cost': solution.cost,
                'backend_used': solution.backend_used,
                'success': True,
                'message': 'Tasks assigned successfully'
            }
            
        except Exception as e:
            return {
                'assignments': {},
                'completion_time': 0,
                'total_cost': 0,
                'backend_used': 'none',
                'success': False,
                'error': str(e),
                'message': f'Assignment failed: {e}'
            }


def quick_assign(agents: List[Dict], tasks: List[Dict], minimize: str = "time") -> Dict:
    """
    One-liner function for instant task assignment.
    
    Args:
        agents: List of agent dicts
        tasks: List of task dicts  
        minimize: 'time' or 'cost'
        
    Returns:
        Assignment result dict
        
    Example:
        >>> result = quick_assign(
        ...     agents=[{'id': 'dev1', 'skills': ['python'], 'capacity': 2}],
        ...     tasks=[{'id': 'api', 'skills': ['python'], 'duration': 4}]
        ... )
    """
    planner = SimpleQuantumPlanner()
    return planner.assign_tasks(agents, tasks, minimize)


def demo_simple_usage():
    """Demonstrate simple usage patterns."""
    print("🚀 Simple Quantum Task Planner Demo")
    print("=" * 50)
    
    # Example 1: Basic software team
    agents = [
        {'id': 'alice', 'skills': ['python', 'ml'], 'capacity': 3},
        {'id': 'bob', 'skills': ['javascript', 'react'], 'capacity': 2},
        {'id': 'charlie', 'skills': ['python', 'devops'], 'capacity': 2}
    ]
    
    tasks = [
        {'id': 'api_backend', 'skills': ['python'], 'priority': 5, 'duration': 2},
        {'id': 'react_frontend', 'skills': ['javascript', 'react'], 'priority': 3, 'duration': 3},
        {'id': 'ml_model', 'skills': ['python', 'ml'], 'priority': 8, 'duration': 4},
        {'id': 'deployment', 'skills': ['devops'], 'priority': 6, 'duration': 1}
    ]
    
    print("\n📋 Software Development Team Assignment")
    result = quick_assign(agents, tasks, minimize="time")
    
    if result['success']:
        print(f"✅ Assignment successful in {result['completion_time']} time units")
        print(f"🔧 Backend used: {result['backend_used']}")
        print(f"📊 Assignments:")
        for task, agent in result['assignments'].items():
            print(f"   {task} → {agent}")
    else:
        print(f"❌ Assignment failed: {result['error']}")
    
    # Example 2: Research team
    print("\n🔬 Research Team Assignment")
    research_agents = [
        {'id': 'researcher_a', 'skills': ['quantum', 'theory'], 'capacity': 1},
        {'id': 'researcher_b', 'skills': ['machine_learning', 'implementation'], 'capacity': 2},
        {'id': 'researcher_c', 'skills': ['quantum', 'implementation'], 'capacity': 2}
    ]
    
    research_tasks = [
        {'id': 'quantum_algorithm', 'skills': ['quantum', 'theory'], 'priority': 10, 'duration': 5},
        {'id': 'ml_integration', 'skills': ['machine_learning'], 'priority': 7, 'duration': 3},
        {'id': 'implementation', 'skills': ['implementation'], 'priority': 8, 'duration': 4}
    ]
    
    result2 = quick_assign(research_agents, research_tasks, minimize="time")
    
    if result2['success']:
        print(f"✅ Research assignment successful in {result2['completion_time']} time units")
        print(f"📊 Assignments:")
        for task, agent in result2['assignments'].items():
            print(f"   {task} → {agent}")
    else:
        print(f"❌ Research assignment failed: {result2['error']}")


if __name__ == "__main__":
    demo_simple_usage()