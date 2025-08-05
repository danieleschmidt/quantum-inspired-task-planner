"""Command-line interface for Quantum Task Planner."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .planner import QuantumTaskPlanner, PlannerConfig
from .models import Agent, Task, TimeWindowTask
from .optimizer import OptimizationBackend


console = Console()


def load_problem_file(file_path: Path) -> Dict[str, Any]:
    """Load problem definition from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: File {file_path} not found[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in {file_path}: {e}[/red]")
        sys.exit(1)


def parse_agents(agents_data: List[Dict[str, Any]]) -> List[Agent]:
    """Parse agents from JSON data."""
    agents = []
    for agent_data in agents_data:
        agents.append(Agent(
            id=agent_data["id"],
            skills=agent_data.get("skills", []),
            capacity=agent_data.get("capacity", 1),
            cost_per_hour=agent_data.get("cost_per_hour", 0.0),
            availability=agent_data.get("availability", {})
        ))
    return agents


def parse_tasks(tasks_data: List[Dict[str, Any]], time_windows: bool = False) -> List[Task]:
    """Parse tasks from JSON data."""
    tasks = []
    for task_data in tasks_data:
        if time_windows and any(key in task_data for key in ["earliest_start", "latest_finish"]):
            tasks.append(TimeWindowTask(
                id=task_data["id"],
                required_skills=task_data.get("required_skills", []),
                priority=task_data.get("priority", 1),
                duration=task_data.get("duration", 1),
                earliest_start=task_data.get("earliest_start", 0),
                latest_finish=task_data.get("latest_finish", float('inf'))
            ))
        else:
            tasks.append(Task(
                id=task_data["id"],
                required_skills=task_data.get("required_skills", []),
                priority=task_data.get("priority", 1),
                duration=task_data.get("duration", 1),
                dependencies=task_data.get("dependencies", [])
            ))
    return tasks


def display_solution(solution, detailed: bool = False):
    """Display solution in a formatted table."""
    if not solution.assignments:
        console.print("[yellow]No assignments found[/yellow]")
        return
    
    # Create assignments table
    table = Table(title="Task Assignments")
    table.add_column("Task", style="cyan")
    table.add_column("Assigned Agent", style="green")
    table.add_column("Duration", style="yellow")
    
    if detailed and hasattr(solution, 'schedule') and solution.schedule:
        table.add_column("Start Time", style="blue")
        table.add_column("End Time", style="blue")
    
    for task_id, agent_id in solution.assignments.items():
        row = [task_id, agent_id]
        
        # Add duration if available
        if hasattr(solution, 'task_durations') and solution.task_durations:
            duration = solution.task_durations.get(task_id, "N/A")
            row.append(str(duration))
        else:
            row.append("N/A")
            
        # Add schedule info if available
        if detailed and hasattr(solution, 'schedule') and solution.schedule:
            agent_schedule = solution.schedule.get(agent_id, {})
            task_schedule = agent_schedule.get(task_id, (None, None))
            start, end = task_schedule
            row.extend([str(start) if start is not None else "N/A", 
                       str(end) if end is not None else "N/A"])
        
        table.add_row(*row)
    
    console.print(table)
    
    # Display metrics
    metrics_panel = Panel.fit(
        f"[bold]Solution Metrics[/bold]\n"
        f"Total Assignments: {len(solution.assignments)}\n"
        f"Makespan: {getattr(solution, 'makespan', 'N/A')}\n"
        f"Total Cost: {getattr(solution, 'total_cost', 'N/A')}\n"
        f"Backend Used: {solution.metadata.get('backend_used', 'N/A') if solution.metadata else 'N/A'}",
        title="📊 Results"
    )
    console.print(metrics_panel)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Quantum Task Planner - Optimize task assignments using quantum and classical algorithms."""
    pass


@main.command()
@click.argument('problem_file', type=click.Path(exists=True, path_type=Path))
@click.option('--backend', '-b', default='auto', 
              type=click.Choice(['auto', 'dwave', 'azure', 'ibm', 'simulator', 'classical']),
              help='Backend to use for optimization')
@click.option('--objective', '-o', default='minimize_makespan',
              type=click.Choice(['minimize_makespan', 'maximize_priority', 'balance_load', 'minimize_cost']),
              help='Optimization objective')
@click.option('--fallback', default='simulated_annealing',
              help='Fallback backend when primary fails')
@click.option('--output', '-f', type=click.Path(path_type=Path),
              help='Output file for solution (JSON format)')
@click.option('--detailed/--no-detailed', default=False,
              help='Show detailed schedule information')
@click.option('--verbose/--quiet', default=False,
              help='Enable verbose output')
def solve(problem_file: Path, backend: str, objective: str, fallback: str, 
          output: Optional[Path], detailed: bool, verbose: bool):
    """Solve a task assignment problem from a JSON file.
    
    The problem file should contain:
    {
        "agents": [{"id": "agent1", "skills": ["python"], "capacity": 2}],
        "tasks": [{"id": "task1", "required_skills": ["python"], "duration": 1}],
        "constraints": {"skill_match": true},
        "time_horizon": 10  // optional for time-windowed problems
    }
    """
    try:
        # Load problem
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Loading problem...", total=None)
            problem_data = load_problem_file(problem_file)
            progress.update(task, description="Parsing problem data...")
            
            # Parse problem components
            agents = parse_agents(problem_data.get("agents", []))
            time_horizon = problem_data.get("time_horizon")
            tasks = parse_tasks(problem_data.get("tasks", []), time_windows=bool(time_horizon))
            constraints = problem_data.get("constraints", {})
            
            progress.update(task, description="Initializing planner...")
            
            # Create planner
            config = PlannerConfig(
                backend=backend,
                fallback=fallback,
                verbose=verbose
            )
            planner = QuantumTaskPlanner(config=config)
            
            progress.update(task, description="Solving optimization problem...")
            
            # Solve problem
            if time_horizon and any(isinstance(t, TimeWindowTask) for t in tasks):
                solution = planner.assign_with_time(
                    agents=agents,
                    tasks=tasks,
                    time_horizon=time_horizon,
                    objective=objective,
                    constraints=constraints
                )
            else:
                solution = planner.assign(
                    agents=agents,
                    tasks=tasks,
                    objective=objective,
                    constraints=constraints
                )
        
        # Display results
        console.print(f"\n[green]✓ Problem solved successfully![/green]")
        display_solution(solution, detailed=detailed)
        
        # Save output if requested
        if output:
            solution_data = {
                "assignments": solution.assignments,
                "makespan": getattr(solution, 'makespan', None),
                "total_cost": getattr(solution, 'total_cost', None),
                "metadata": solution.metadata or {}
            }
            
            if hasattr(solution, 'schedule') and solution.schedule:
                solution_data["schedule"] = solution.schedule
            
            with open(output, 'w') as f:
                json.dump(solution_data, f, indent=2)
            console.print(f"[green]Solution saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error solving problem: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option('--agents', '-a', default=5, type=int, help='Number of agents')
@click.option('--tasks', '-t', default=10, type=int, help='Number of tasks')
@click.option('--skills', '-s', default=3, type=int, help='Number of different skills')
@click.option('--output', '-o', type=click.Path(path_type=Path), required=True,
              help='Output file for generated problem')
@click.option('--time-windows/--no-time-windows', default=False,
              help='Generate time-windowed tasks')
def generate(agents: int, tasks: int, skills: int, output: Path, time_windows: bool):
    """Generate a random task assignment problem."""
    import random
    
    skill_names = [f"skill_{i}" for i in range(skills)]
    
    # Generate agents
    agents_data = []
    for i in range(agents):
        agent_skills = random.sample(skill_names, random.randint(1, min(3, skills)))
        agents_data.append({
            "id": f"agent_{i}",
            "skills": agent_skills,
            "capacity": random.randint(1, 3),
            "cost_per_hour": round(random.uniform(50, 150), 2)
        })
    
    # Generate tasks
    tasks_data = []
    for i in range(tasks):
        required_skills = random.sample(skill_names, random.randint(1, 2))
        task_data = {
            "id": f"task_{i}",
            "required_skills": required_skills,
            "priority": random.randint(1, 10),
            "duration": random.randint(1, 5)
        }
        
        if time_windows:
            earliest = random.randint(0, 10)
            task_data.update({
                "earliest_start": earliest,
                "latest_finish": earliest + random.randint(5, 15)
            })
        
        tasks_data.append(task_data)
    
    # Create problem structure
    problem = {
        "agents": agents_data,
        "tasks": tasks_data,
        "constraints": {
            "skill_match": True,
            "capacity_limit": True
        }
    }
    
    if time_windows:
        problem["time_horizon"] = 20
    
    # Save to file
    with open(output, 'w') as f:
        json.dump(problem, f, indent=2)
    
    console.print(f"[green]Generated problem with {agents} agents and {tasks} tasks[/green]")
    console.print(f"[green]Saved to {output}[/green]")


@main.command()
@click.option('--backend', '-b', default='auto',
              type=click.Choice(['auto', 'dwave', 'azure', 'ibm', 'simulator', 'classical']),
              help='Backend to check')
def status(backend: str):
    """Check the status of quantum backends."""
    try:
        config = PlannerConfig(backend=backend, verbose=True)
        planner = QuantumTaskPlanner(config=config)
        
        properties = planner.get_device_properties()
        
        status_table = Table(title=f"Backend Status: {backend}")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="green")
        
        for key, value in properties.items():
            status_table.add_row(str(key), str(value))
        
        console.print(status_table)
        
    except Exception as e:
        console.print(f"[red]Error checking backend status: {e}[/red]")
        sys.exit(1)


@main.command()
def backends():
    """List available backends and their capabilities."""
    backends_info = [
        ("Auto", "Automatically selects best available backend", "✓"),
        ("D-Wave", "Quantum annealing using D-Wave systems", "?" ),
        ("Azure Quantum", "Microsoft Azure Quantum cloud service", "?"),
        ("IBM Quantum", "IBM Quantum cloud and simulators", "?"),
        ("Simulator", "Local quantum simulator", "✓"),
        ("Classical", "Simulated annealing (always available)", "✓"),
    ]
    
    table = Table(title="Available Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Available", style="green")
    
    for name, desc, available in backends_info:
        table.add_row(name, desc, available)
    
    console.print(table)
    console.print("\n[yellow]Note: ? indicates backend requires credentials and may not be available[/yellow]")


if __name__ == "__main__":
    main()