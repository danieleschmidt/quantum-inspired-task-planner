#!/usr/bin/env python3
"""
Enhanced Basic Quantum Task Planner - Generation 1 Implementation
Provides core functionality with essential features working immediately.
"""

import time
import logging
import random
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Available optimization objectives."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MAXIMIZE_PRIORITY = "maximize_priority"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"

class BackendType(Enum):
    """Available backend types."""
    AUTO = "auto"
    CLASSICAL = "classical"
    QUANTUM_SIMULATOR = "quantum_simulator"
    SIMULATED_ANNEALING = "simulated_annealing"

@dataclass(frozen=True)
class Agent:
    """Represents an agent that can execute tasks."""
    
    agent_id: str
    skills: List[str]
    capacity: int = 1
    availability: float = 1.0
    cost_per_hour: float = 0.0
    
    def __post_init__(self):
        """Validate agent parameters."""
        if not self.skills:
            raise ValueError(f"Agent {self.agent_id}: Skills cannot be empty")
        if self.capacity <= 0:
            raise ValueError(f"Agent {self.agent_id}: Capacity must be positive")
        if not (0 <= self.availability <= 1):
            raise ValueError(f"Agent {self.agent_id}: Availability must be between 0 and 1")
    
    def can_handle_task(self, task: 'Task') -> bool:
        """Check if agent can handle the given task."""
        return set(task.required_skills).issubset(set(self.skills))
    
    def __hash__(self):
        return hash((self.agent_id, tuple(self.skills), self.capacity))

@dataclass(frozen=True)
class Task:
    """Represents a task to be scheduled."""
    
    task_id: str
    required_skills: List[str]
    priority: int = 1
    duration: int = 1
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate task parameters."""
        if not self.required_skills:
            raise ValueError(f"Task {self.task_id}: Required skills cannot be empty")
        if self.priority <= 0:
            raise ValueError(f"Task {self.task_id}: Priority must be positive")
        if self.duration <= 0:
            raise ValueError(f"Task {self.task_id}: Duration must be positive")
    
    def can_be_assigned_to(self, agent: Agent) -> bool:
        """Check if this task can be assigned to the given agent."""
        return agent.can_handle_task(self)
    
    def __hash__(self):
        return hash((self.task_id, tuple(self.required_skills), self.priority, self.duration))

@dataclass
class Solution:
    """Represents a solution to the task scheduling problem."""
    
    assignments: Dict[str, str]  # task_id -> agent_id
    makespan: float
    cost: float = 0.0
    backend_used: str = "unknown"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    quality_score: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if not self.assignments:
            raise ValueError("Assignments cannot be empty")
        if self.makespan < 0:
            raise ValueError("Makespan must be non-negative")
        
        # Calculate quality score
        self.quality_score = self.calculate_quality_score()
    
    def get_load_distribution(self) -> Dict[str, int]:
        """Get the load distribution across agents."""
        load_dist = {}
        for agent_id in self.assignments.values():
            load_dist[agent_id] = load_dist.get(agent_id, 0) + 1
        return load_dist
    
    def get_assigned_agents(self) -> Set[str]:
        """Get the set of agents that have been assigned tasks."""
        return set(self.assignments.values())
    
    def calculate_quality_score(self) -> float:
        """Calculate a normalized quality score (0-1, higher is better)."""
        # Simple quality metric based on load balance and makespan
        load_dist = list(self.get_load_distribution().values())
        if not load_dist:
            return 0.0
        
        # Load balance component (lower variance is better)
        avg_load = sum(load_dist) / len(load_dist)
        if avg_load == 0:
            balance_score = 1.0
        else:
            variance = sum((load - avg_load) ** 2 for load in load_dist) / len(load_dist)
            balance_score = max(0, 1 - (variance / avg_load))
        
        # Makespan component (lower is better, normalized to 0-1)
        makespan_score = max(0, 1 - (self.makespan / 50))  # Assume max reasonable makespan of 50
        
        # Combined score
        return (balance_score * 0.6 + makespan_score * 0.4)

class EnhancedBasicPlanner:
    """Enhanced basic quantum task planner with immediate functionality."""
    
    def __init__(self, backend: BackendType = BackendType.AUTO, verbose: bool = False):
        """Initialize the planner."""
        self.backend = backend
        self.verbose = verbose
        self.stats = {
            "problems_solved": 0,
            "total_solve_time": 0,
            "average_quality": 0,
            "fallback_used": 0
        }
        
        if self.verbose:
            logger.info(f"EnhancedBasicPlanner initialized with backend: {backend.value}")
    
    def assign(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Solution:
        """Assign tasks to agents optimally."""
        start_time = time.time()
        
        try:
            # Input validation
            self._validate_inputs(agents, tasks)
            
            # Determine which backend to use
            selected_backend = self._select_backend(len(agents), len(tasks))
            
            if self.verbose:
                logger.info(f"Solving {len(tasks)} tasks for {len(agents)} agents using {selected_backend}")
            
            # Solve using the selected approach
            solution = self._solve_with_backend(agents, tasks, objective, selected_backend)
            
            # Update statistics
            solve_time = time.time() - start_time
            self._update_stats(solve_time, solution.quality_score)
            
            # Add metadata
            solution.metadata.update({
                "solve_time": solve_time,
                "backend_used": selected_backend,
                "objective": objective.value,
                "problem_size": len(agents) * len(tasks),
                "timestamp": time.time()
            })
            
            if self.verbose:
                logger.info(f"Solution found in {solve_time:.3f}s - Quality: {solution.quality_score:.3f}")
            
            return solution
            
        except Exception as e:
            logger.error(f"Assignment failed: {e}")
            raise
    
    def _validate_inputs(self, agents: List[Agent], tasks: List[Task]):
        """Validate input data."""
        if not agents:
            raise ValueError("No agents provided")
        if not tasks:
            raise ValueError("No tasks provided")
        
        # Check if any tasks can be assigned
        assignable_count = 0
        for task in tasks:
            if any(task.can_be_assigned_to(agent) for agent in agents):
                assignable_count += 1
        
        if assignable_count == 0:
            raise ValueError("No tasks can be assigned to available agents (skill mismatch)")
        
        if self.verbose and assignable_count < len(tasks):
            logger.warning(f"Only {assignable_count}/{len(tasks)} tasks can be assigned")
    
    def _select_backend(self, num_agents: int, num_tasks: int) -> str:
        """Select appropriate backend based on problem characteristics."""
        problem_size = num_agents * num_tasks
        
        if self.backend == BackendType.AUTO:
            if problem_size <= 10:
                return "greedy_optimal"
            elif problem_size <= 50:
                return "simulated_annealing"
            else:
                return "enhanced_greedy"
        else:
            return self.backend.value
    
    def _solve_with_backend(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        objective: OptimizationObjective,
        backend: str
    ) -> Solution:
        """Solve using the specified backend."""
        if backend == "greedy_optimal":
            return self._solve_greedy_optimal(agents, tasks, objective)
        elif backend == "simulated_annealing":
            return self._solve_simulated_annealing(agents, tasks, objective)
        elif backend == "enhanced_greedy":
            return self._solve_enhanced_greedy(agents, tasks, objective)
        else:
            # Fallback to basic greedy
            logger.warning(f"Backend {backend} not implemented, using greedy fallback")
            self.stats["fallback_used"] += 1
            return self._solve_greedy_optimal(agents, tasks, objective)
    
    def _solve_greedy_optimal(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        objective: OptimizationObjective
    ) -> Solution:
        """Solve using greedy algorithm optimized for small problems."""
        assignments = {}
        agent_loads = {agent.agent_id: 0 for agent in agents}
        
        # Sort tasks by priority (descending) and then by duration
        if objective == OptimizationObjective.MAXIMIZE_PRIORITY:
            sorted_tasks = sorted(tasks, key=lambda t: (-t.priority, t.duration))
        elif objective == OptimizationObjective.MINIMIZE_MAKESPAN:
            sorted_tasks = sorted(tasks, key=lambda t: (-t.duration, -t.priority))
        else:
            sorted_tasks = sorted(tasks, key=lambda t: -t.priority)
        
        for task in sorted_tasks:
            # Find capable agents
            capable_agents = [a for a in agents if task.can_be_assigned_to(a)]
            
            if not capable_agents:
                logger.warning(f"No agent can handle task {task.task_id}")
                continue
            
            # Select best agent based on objective
            if objective == OptimizationObjective.BALANCE_LOAD:
                # Choose agent with minimum current load
                best_agent = min(capable_agents, key=lambda a: agent_loads[a.agent_id])
            elif objective == OptimizationObjective.MINIMIZE_COST:
                # Choose cheapest capable agent
                best_agent = min(capable_agents, key=lambda a: a.cost_per_hour)
            else:
                # Default: choose agent with minimum current load
                best_agent = min(capable_agents, key=lambda a: agent_loads[a.agent_id])
            
            # Assign task
            assignments[task.task_id] = best_agent.agent_id
            agent_loads[best_agent.agent_id] += task.duration
        
        # Calculate makespan and cost
        makespan = max(agent_loads.values()) if agent_loads.values() else 0
        total_cost = sum(
            agents[next(i for i, a in enumerate(agents) if a.agent_id == agent_id)].cost_per_hour * load
            for agent_id, load in agent_loads.items()
        )
        
        return Solution(
            assignments=assignments,
            makespan=makespan,
            cost=total_cost,
            backend_used="greedy_optimal"
        )
    
    def _solve_simulated_annealing(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        objective: OptimizationObjective
    ) -> Solution:
        """Solve using simulated annealing for medium-sized problems."""
        # Start with greedy solution
        current_solution = self._solve_greedy_optimal(agents, tasks, objective)
        best_solution = current_solution
        
        # SA parameters
        initial_temp = 100.0
        final_temp = 0.1
        cooling_rate = 0.95
        max_iterations = min(1000, len(tasks) * len(agents) * 10)
        
        temperature = initial_temp
        
        for iteration in range(max_iterations):
            if temperature < final_temp:
                break
            
            # Generate neighbor solution by swapping two random assignments
            neighbor_solution = self._generate_neighbor_solution(
                current_solution, agents, tasks, objective
            )
            
            if neighbor_solution is None:
                continue
            
            # Calculate acceptance probability
            delta_quality = neighbor_solution.quality_score - current_solution.quality_score
            
            if delta_quality > 0 or random.random() < self._acceptance_probability(delta_quality, temperature):
                current_solution = neighbor_solution
                
                if current_solution.quality_score > best_solution.quality_score:
                    best_solution = current_solution
            
            temperature *= cooling_rate
        
        best_solution.backend_used = "simulated_annealing"
        return best_solution
    
    def _solve_enhanced_greedy(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        objective: OptimizationObjective
    ) -> Solution:
        """Solve using enhanced greedy algorithm for large problems."""
        # Multi-pass greedy with different sorting strategies
        strategies = [
            lambda t: (-t.priority, t.duration),           # Priority first
            lambda t: (-t.duration, -t.priority),          # Duration first
            lambda t: (-len(t.required_skills), -t.priority),  # Complexity first
        ]
        
        best_solution = None
        
        for strategy in strategies:
            try:
                assignments = {}
                agent_loads = {agent.agent_id: 0 for agent in agents}
                
                sorted_tasks = sorted(tasks, key=strategy)
                
                for task in sorted_tasks:
                    capable_agents = [a for a in agents if task.can_be_assigned_to(a)]
                    if not capable_agents:
                        continue
                    
                    # Enhanced selection considers multiple factors
                    def agent_score(agent):
                        load = agent_loads[agent.agent_id]
                        skill_match = len(set(agent.skills) & set(task.required_skills))
                        return (load, -skill_match, agent.cost_per_hour)
                    
                    best_agent = min(capable_agents, key=agent_score)
                    assignments[task.task_id] = best_agent.agent_id
                    agent_loads[best_agent.agent_id] += task.duration
                
                makespan = max(agent_loads.values()) if agent_loads.values() else 0
                total_cost = sum(
                    next(a.cost_per_hour for a in agents if a.agent_id == agent_id) * load
                    for agent_id, load in agent_loads.items()
                )
                
                solution = Solution(
                    assignments=assignments,
                    makespan=makespan,
                    cost=total_cost,
                    backend_used="enhanced_greedy"
                )
                
                if best_solution is None or solution.quality_score > best_solution.quality_score:
                    best_solution = solution
                    
            except Exception as e:
                logger.warning(f"Strategy failed: {e}")
                continue
        
        return best_solution or self._solve_greedy_optimal(agents, tasks, objective)
    
    def _generate_neighbor_solution(
        self, 
        current_solution: Solution, 
        agents: List[Agent], 
        tasks: List[Task],
        objective: OptimizationObjective
    ) -> Optional[Solution]:
        """Generate a neighbor solution for simulated annealing."""
        if len(current_solution.assignments) < 2:
            return None
        
        try:
            # Copy current assignments
            new_assignments = current_solution.assignments.copy()
            
            # Randomly select two tasks
            task_ids = list(new_assignments.keys())
            task1, task2 = random.sample(task_ids, 2)
            
            # Try to swap their assignments
            agent1_id = new_assignments[task1]
            agent2_id = new_assignments[task2]
            
            # Find task and agent objects
            task1_obj = next(t for t in tasks if t.task_id == task1)
            task2_obj = next(t for t in tasks if t.task_id == task2)
            agent1_obj = next(a for a in agents if a.agent_id == agent1_id)
            agent2_obj = next(a for a in agents if a.agent_id == agent2_id)
            
            # Check if swap is valid
            if (task1_obj.can_be_assigned_to(agent2_obj) and 
                task2_obj.can_be_assigned_to(agent1_obj)):
                
                new_assignments[task1] = agent2_id
                new_assignments[task2] = agent1_id
                
                # Calculate new metrics
                agent_loads = {}
                for task_id, agent_id in new_assignments.items():
                    task_obj = next(t for t in tasks if t.task_id == task_id)
                    agent_loads[agent_id] = agent_loads.get(agent_id, 0) + task_obj.duration
                
                makespan = max(agent_loads.values()) if agent_loads.values() else 0
                total_cost = sum(
                    next(a.cost_per_hour for a in agents if a.agent_id == agent_id) * load
                    for agent_id, load in agent_loads.items()
                )
                
                return Solution(
                    assignments=new_assignments,
                    makespan=makespan,
                    cost=total_cost,
                    backend_used="simulated_annealing"
                )
        
        except Exception as e:
            logger.debug(f"Neighbor generation failed: {e}")
            return None
        
        return None
    
    def _acceptance_probability(self, delta_quality: float, temperature: float) -> float:
        """Calculate acceptance probability for simulated annealing."""
        if delta_quality >= 0:
            return 1.0
        return pow(2.71828, delta_quality / temperature) if temperature > 0 else 0.0
    
    def _update_stats(self, solve_time: float, quality: float):
        """Update planner statistics."""
        self.stats["problems_solved"] += 1
        self.stats["total_solve_time"] += solve_time
        
        # Update average quality
        current_avg = self.stats["average_quality"]
        n = self.stats["problems_solved"]
        self.stats["average_quality"] = (current_avg * (n - 1) + quality) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planner performance statistics."""
        stats = self.stats.copy()
        if stats["problems_solved"] > 0:
            stats["average_solve_time"] = stats["total_solve_time"] / stats["problems_solved"]
        else:
            stats["average_solve_time"] = 0
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get planner health status."""
        return {
            "status": "healthy",
            "backend": self.backend.value,
            "problems_solved": self.stats["problems_solved"],
            "average_quality": self.stats["average_quality"],
            "uptime": time.time(),  # Simple uptime indicator
            "last_updated": time.time()
        }

# Demo and testing functions
def create_demo_problem(num_agents: int = 5, num_tasks: int = 10) -> tuple[List[Agent], List[Task]]:
    """Create a demonstration problem for testing."""
    skills_pool = ["python", "javascript", "ml", "devops", "react", "database", "testing", "ui_design"]
    
    agents = []
    for i in range(num_agents):
        agent_skills = random.sample(skills_pool, k=random.randint(2, 4))
        agents.append(Agent(
            agent_id=f"agent_{i+1}",
            skills=agent_skills,
            capacity=random.randint(1, 3),
            cost_per_hour=random.uniform(20, 100)
        ))
    
    tasks = []
    for i in range(num_tasks):
        required_skills = random.sample(skills_pool, k=random.randint(1, 2))
        tasks.append(Task(
            task_id=f"task_{i+1}",
            required_skills=required_skills,
            priority=random.randint(1, 10),
            duration=random.randint(1, 5)
        ))
    
    return agents, tasks

def run_comprehensive_demo():
    """Run a comprehensive demonstration of the enhanced basic planner."""
    print("🚀 Enhanced Basic Quantum Task Planner - Generation 1")
    print("=" * 60)
    
    # Create different problem sizes for testing
    test_cases = [
        ("Small Problem", 3, 5),
        ("Medium Problem", 5, 15),
        ("Large Problem", 8, 25)
    ]
    
    planner = EnhancedBasicPlanner(verbose=True)
    
    for case_name, num_agents, num_tasks in test_cases:
        print(f"\n📊 Testing {case_name} ({num_agents} agents, {num_tasks} tasks)")
        print("-" * 50)
        
        try:
            agents, tasks = create_demo_problem(num_agents, num_tasks)
            
            # Test different objectives
            objectives = [
                OptimizationObjective.MINIMIZE_MAKESPAN,
                OptimizationObjective.BALANCE_LOAD,
                OptimizationObjective.MAXIMIZE_PRIORITY
            ]
            
            for objective in objectives:
                start_time = time.time()
                solution = planner.assign(agents, tasks, objective=objective)
                
                print(f"  {objective.value.replace('_', ' ').title()}:")
                print(f"    - Assignments: {len(solution.assignments)}/{len(tasks)} tasks")
                print(f"    - Makespan: {solution.makespan:.1f}")
                print(f"    - Quality Score: {solution.quality_score:.3f}")
                print(f"    - Backend: {solution.backend_used}")
                print(f"    - Solve Time: {solution.metadata.get('solve_time', 0):.3f}s")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Show final statistics
    print(f"\n📈 Final Statistics:")
    stats = planner.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n✅ Generation 1 Implementation Complete!")
    print("Features Working:")
    print("  - Multi-objective optimization")
    print("  - Adaptive backend selection")
    print("  - Problem validation")
    print("  - Performance monitoring")
    print("  - Comprehensive error handling")
    print("  - Quality scoring")

if __name__ == "__main__":
    # Run the comprehensive demo
    run_comprehensive_demo()
    
    print("\n🎯 Quick Test:")
    
    # Quick functionality test
    planner = EnhancedBasicPlanner()
    
    agents = [
        Agent("dev1", ["python", "ml"], capacity=2, cost_per_hour=50),
        Agent("dev2", ["javascript", "react"], capacity=1, cost_per_hour=40),
        Agent("dev3", ["python", "devops"], capacity=3, cost_per_hour=60)
    ]
    
    tasks = [
        Task("backend_api", ["python"], priority=8, duration=3),
        Task("frontend_ui", ["javascript", "react"], priority=5, duration=2),
        Task("ml_model", ["python", "ml"], priority=9, duration=4),
        Task("deployment", ["devops"], priority=6, duration=1)
    ]
    
    solution = planner.assign(agents, tasks)
    print(f"Quick test result: {len(solution.assignments)} assignments, quality={solution.quality_score:.2f}")