"""Basic usage example matching the README documentation."""

from quantum_planner import QuantumTaskPlanner, Agent, Task

def basic_task_assignment():
    """Basic task assignment example from README."""
    
    # Initialize planner
    planner = QuantumTaskPlanner(
        backend="auto",  # Automatically select best backend
        fallback="simulated_annealing"
    )

    # Define agents with skills and capacity
    agents = [
        Agent("agent1", skills=["python", "ml"], capacity=3),
        Agent("agent2", skills=["javascript", "react"], capacity=2),
        Agent("agent3", skills=["python", "devops"], capacity=2),
    ]

    # Define tasks with requirements
    tasks = [
        Task("backend_api", required_skills=["python"], priority=5, duration=2),
        Task("frontend_ui", required_skills=["javascript", "react"], priority=3, duration=3),
        Task("ml_pipeline", required_skills=["python", "ml"], priority=8, duration=4),
        Task("deployment", required_skills=["devops"], priority=6, duration=1),
    ]

    # Solve assignment problem
    solution = planner.assign(
        agents=agents,
        tasks=tasks,
        objective="minimize_makespan",  # or "maximize_priority", "balance_load"
        constraints={
            "skill_match": True,
            "capacity_limit": True,
            "precedence": {"ml_pipeline": ["backend_api"]}
        }
    )

    print(f"Assignments: {solution.assignments}")
    print(f"Makespan: {solution.makespan}")
    print(f"Solver used: {solution.metadata.get('backend_used', 'unknown')}")
    
    return solution


def time_window_example():
    """Time window example from README."""
    from quantum_planner import TimeWindowTask
    
    planner = QuantumTaskPlanner(backend="auto")
    
    # Define agents
    agents = [
        Agent("agent1", skills=["python"], capacity=2),
        Agent("agent2", skills=["devops"], capacity=1),
    ]

    # Tasks with time constraints
    tasks = [
        TimeWindowTask(
            "urgent_fix",
            required_skills=["python"],
            earliest_start=0,
            latest_finish=4,
            duration=2
        ),
        TimeWindowTask(
            "scheduled_maintenance",
            required_skills=["devops"],
            earliest_start=10,
            latest_finish=15,
            duration=3
        ),
    ]

    # Solve with temporal constraints
    solution = planner.assign_with_time(
        agents=agents,
        tasks=tasks,
        time_horizon=20
    )

    # Get schedule
    print("Schedule:")
    if hasattr(solution, 'schedule') and solution.schedule:
        for agent, schedule in solution.schedule.items():
            print(f"{agent}:")
            for task, (start, end) in schedule.items():
                print(f"  {task}: {start}-{end}")
    else:
        print("Schedule information not available")
        print(f"Assignments: {solution.assignments}")
    
    return solution


def multiple_objectives_example():
    """Example with different optimization objectives."""
    
    planner = QuantumTaskPlanner(backend="auto")
    
    agents = [
        Agent("agent1", skills=["python", "data"], capacity=2, cost_per_hour=100),
        Agent("agent2", skills=["python", "web"], capacity=3, cost_per_hour=80),
        Agent("agent3", skills=["data", "ml"], capacity=2, cost_per_hour=120),
    ]
    
    tasks = [
        Task("data_pipeline", required_skills=["data"], priority=7, duration=3),
        Task("web_api", required_skills=["web"], priority=5, duration=2),
        Task("ml_model", required_skills=["ml"], priority=9, duration=4),
        Task("testing", required_skills=["python"], priority=4, duration=1),
    ]
    
    objectives = ["minimize_makespan", "maximize_priority", "balance_load", "minimize_cost"]
    
    print("Comparing different optimization objectives:")
    print("=" * 50)
    
    for objective in objectives:
        solution = planner.assign(
            agents=agents,
            tasks=tasks,
            objective=objective,
            constraints={"skill_match": True, "capacity_limit": True}
        )
        
        print(f"\nObjective: {objective}")
        print(f"Assignments: {solution.assignments}")
        print(f"Makespan: {getattr(solution, 'makespan', 'N/A')}")
        print(f"Total Cost: {getattr(solution, 'total_cost', 'N/A')}")
        print(f"Backend: {solution.metadata.get('backend_used', 'unknown')}")
    
    return solution


def error_handling_example():
    """Example showing error handling and fallback behavior."""
    
    # Try to use a quantum backend that might not be available
    planner = QuantumTaskPlanner(
        backend="dwave",  # Might not be available
        fallback="simulated_annealing",
        config=None
    )
    
    agents = [Agent("agent1", skills=["python"], capacity=1)]
    tasks = [Task("simple_task", required_skills=["python"], duration=1)]
    
    try:
        solution = planner.assign(agents, tasks)
        print("Solution found:")
        print(f"Assignments: {solution.assignments}")
        print(f"Backend used: {solution.metadata.get('backend_used', 'unknown')}")
        print(f"Fallback used: {solution.metadata.get('fallback_used', False)}")
        
        if solution.metadata.get('fallback_used'):
            print(f"Primary backend error: {solution.metadata.get('primary_backend_error', 'N/A')}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return None


if __name__ == "__main__":
    print("Quantum Task Planner - Basic Usage Examples")
    print("=" * 50)
    
    print("\n1. Basic Task Assignment:")
    basic_task_assignment()
    
    print("\n2. Time Window Constraints:")
    time_window_example()
    
    print("\n3. Multiple Objectives:")
    multiple_objectives_example()
    
    print("\n4. Error Handling:")
    error_handling_example()
    
    print("\nAll examples completed!")