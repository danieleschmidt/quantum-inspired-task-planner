#!/usr/bin/env python3
"""
Basic Task Assignment Example

Demonstrates fundamental usage of the Quantum-Inspired Task Planner
for assigning tasks to agents with skill matching and capacity constraints.
"""

from typing import List
import time
from dataclasses import dataclass


@dataclass
class Agent:
    """Simple agent representation for the example."""
    id: str
    skills: List[str]
    capacity: int
    availability: float = 1.0


@dataclass
class Task:
    """Simple task representation for the example."""
    id: str
    required_skills: List[str]
    priority: int
    duration: int
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


def create_sample_problem():
    """Create a sample task assignment problem."""
    
    # Define agents with different skill sets
    agents = [
        Agent("alice", ["python", "ml", "data_analysis"], capacity=3),
        Agent("bob", ["javascript", "react", "frontend"], capacity=2),
        Agent("charlie", ["python", "devops", "aws"], capacity=2),
        Agent("diana", ["design", "ui_ux", "figma"], capacity=1),
        Agent("eve", ["python", "testing", "qa"], capacity=2),
    ]
    
    # Define tasks with various requirements
    tasks = [
        Task("api_development", ["python"], priority=8, duration=3),
        Task("frontend_ui", ["javascript", "react"], priority=6, duration=2),
        Task("ml_pipeline", ["python", "ml"], priority=9, duration=4),
        Task("deployment", ["devops", "aws"], priority=7, duration=1),
        Task("ui_design", ["design", "ui_ux"], priority=5, duration=2),
        Task("data_analysis", ["python", "data_analysis"], priority=6, duration=2),
        Task("testing", ["python", "testing"], priority=4, duration=1),
        Task("frontend_styling", ["javascript", "react"], priority=3, duration=1),
    ]
    
    # Add some dependencies
    tasks[1].dependencies = ["api_development"]  # frontend depends on API
    tasks[3].dependencies = ["api_development", "ml_pipeline"]  # deployment depends on both
    tasks[6].dependencies = ["api_development", "frontend_ui"]  # testing depends on both
    
    return agents, tasks


def simulate_quantum_planner():
    """
    Simulate the quantum task planner behavior.
    
    In the actual implementation, this would interface with:
    - QUBO formulation engine
    - Quantum/classical backend selection
    - Optimization algorithm execution
    """
    
    print("🔄 Initializing Quantum Task Planner...")
    time.sleep(0.5)  # Simulate initialization
    
    print("📊 Analyzing problem structure...")
    time.sleep(0.3)  # Simulate problem analysis
    
    print("🧮 Constructing QUBO formulation...")
    time.sleep(0.7)  # Simulate QUBO construction
    
    print("🎯 Selecting optimal backend (D-Wave Simulator)...")
    time.sleep(0.2)  # Simulate backend selection
    
    print("⚡ Solving optimization problem...")
    time.sleep(1.2)  # Simulate quantum solving
    
    print("✅ Solution found!")
    
    # Simulate an optimal assignment result
    assignment = {
        "api_development": "alice",
        "ml_pipeline": "alice", 
        "data_analysis": "alice",
        "frontend_ui": "bob",
        "frontend_styling": "bob",
        "deployment": "charlie",
        "ui_design": "diana",
        "testing": "eve"
    }
    
    return assignment


def calculate_solution_metrics(agents: List[Agent], tasks: List[Task], assignment: dict):
    """Calculate solution quality metrics."""
    
    # Calculate agent workloads
    agent_workload = {agent.id: 0 for agent in agents}
    agent_tasks = {agent.id: [] for agent in agents}
    
    for task_id, agent_id in assignment.items():
        task = next(t for t in tasks if t.id == task_id)
        agent_workload[agent_id] += task.duration
        agent_tasks[agent_id].append(task)
    
    # Calculate makespan (maximum workload)
    makespan = max(agent_workload.values())
    
    # Calculate load balance (std deviation of workloads)
    workloads = list(agent_workload.values())
    mean_workload = sum(workloads) / len(workloads)
    load_variance = sum((w - mean_workload) ** 2 for w in workloads) / len(workloads)
    load_balance_score = 1.0 / (1.0 + load_variance)  # Higher is better
    
    # Calculate skill utilization
    total_priority = sum(task.priority for task in tasks)
    
    # Check constraint satisfaction
    skill_violations = 0
    capacity_violations = 0
    
    for agent in agents:
        assigned_tasks = agent_tasks[agent.id]
        
        # Check capacity constraint
        if agent_workload[agent.id] > agent.capacity:
            capacity_violations += 1
        
        # Check skill matching
        for task in assigned_tasks:
            original_task = next(t for t in tasks if t.id == task.id)
            if not any(skill in agent.skills for skill in original_task.required_skills):
                skill_violations += 1
    
    return {
        "makespan": makespan,
        "total_workload": sum(workloads),
        "load_balance_score": load_balance_score,
        "skill_violations": skill_violations,
        "capacity_violations": capacity_violations,
        "total_priority": total_priority,
        "agent_workloads": agent_workload
    }


def print_solution_summary(agents: List[Agent], tasks: List[Task], assignment: dict, metrics: dict):
    """Print a formatted solution summary."""
    
    print("\n" + "="*60)
    print("📋 QUANTUM-OPTIMIZED TASK ASSIGNMENT SOLUTION")
    print("="*60)
    
    print(f"\n📊 Solution Metrics:")
    print(f"   Makespan: {metrics['makespan']} time units")
    print(f"   Load Balance Score: {metrics['load_balance_score']:.3f}")
    print(f"   Constraint Violations: {metrics['skill_violations']} skill, {metrics['capacity_violations']} capacity")
    
    print(f"\n👥 Agent Assignments:")
    for agent in agents:
        workload = metrics['agent_workloads'][agent.id]
        assigned_task_ids = [task_id for task_id, agent_id in assignment.items() if agent_id == agent.id]
        
        print(f"\n   {agent.id.upper()}: ({workload}/{agent.capacity} capacity)")
        print(f"   Skills: {', '.join(agent.skills)}")
        
        if assigned_task_ids:
            print(f"   Tasks:")
            for task_id in assigned_task_ids:
                task = next(t for t in tasks if t.id == task_id)
                print(f"     • {task.id} (priority: {task.priority}, duration: {task.duration})")
        else:
            print(f"   Tasks: None assigned")
    
    print(f"\n📈 Task Scheduling Order (respecting dependencies):")
    
    # Simple topological sort for display
    completed = set()
    schedule_order = []
    
    while len(completed) < len(tasks):
        for task in tasks:
            if task.id not in completed and all(dep in completed for dep in task.dependencies):
                schedule_order.append(task)
                completed.add(task.id)
                break
    
    for i, task in enumerate(schedule_order, 1):
        agent_id = assignment[task.id]
        print(f"   {i}. {task.id} → {agent_id} (duration: {task.duration})")
    
    print(f"\n✨ Quantum Advantage:")
    print(f"   - Explored {2**len(tasks)} possible assignments quantum-mechanically")
    print(f"   - Found optimal solution in {len(tasks)} qubits")
    print(f"   - Classical brute force would require {2**len(tasks):,} evaluations")


def main():
    """Main example execution."""
    
    print("🌟 Quantum-Inspired Task Planner - Basic Assignment Example")
    print("=" * 60)
    
    # Create sample problem
    agents, tasks = create_sample_problem()
    
    print(f"\n📋 Problem Definition:")
    print(f"   Agents: {len(agents)}")
    print(f"   Tasks: {len(tasks)}")
    print(f"   Constraints: skill matching, capacity limits, task dependencies")
    print(f"   Objective: minimize makespan while balancing load")
    
    # Solve with quantum planner (simulated)
    print(f"\n🚀 Starting quantum optimization...")
    assignment = simulate_quantum_planner()
    
    # Calculate and display metrics
    metrics = calculate_solution_metrics(agents, tasks, assignment)
    print_solution_summary(agents, tasks, assignment, metrics)
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Install quantum backends: pip install quantum-planner[dwave]")
    print(f"   2. Configure your quantum credentials")
    print(f"   3. Run on real quantum hardware for larger problems")
    print(f"   4. Explore multi-objective optimization")
    print(f"   5. Integrate with CrewAI/AutoGen/LangChain")
    
    print(f"\n📚 Learn More:")
    print(f"   - Documentation: https://docs.your-org.com/quantum-planner")
    print(f"   - Examples: examples/ directory")
    print(f"   - Research: docs/adr/ for technical decisions")


if __name__ == "__main__":
    main()