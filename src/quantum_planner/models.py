"""Core data models for the quantum task planner."""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from collections import Counter


@dataclass(frozen=True)
class Agent:
    """Represents an agent that can execute tasks."""
    
    agent_id: str
    skills: List[str]
    capacity: int
    availability: float = 1.0
    preferences: Dict[str, Any] = field(default_factory=dict)
    cost_per_hour: float = 0.0
    
    def __init__(self, id: str = None, agent_id: str = None, skills: List[str] = None, 
                 capacity: int = 1, availability: float = 1.0, 
                 preferences: Dict[str, Any] = None, cost_per_hour: float = 0.0):
        """Initialize agent with flexible parameter names."""
        # Handle both 'id' and 'agent_id' parameter names
        if id is not None and agent_id is None:
            agent_id = id
        elif agent_id is None and id is None:
            raise ValueError("Either 'id' or 'agent_id' must be provided")
            
        object.__setattr__(self, 'agent_id', agent_id)
        object.__setattr__(self, 'skills', skills or [])
        object.__setattr__(self, 'capacity', capacity)
        object.__setattr__(self, 'availability', availability)
        object.__setattr__(self, 'preferences', preferences or {})
        object.__setattr__(self, 'cost_per_hour', cost_per_hour)
        
        self.__post_init__()
    
    @property
    def id(self) -> str:
        """Alias for agent_id for API compatibility."""
        return self.agent_id
    
    def __post_init__(self):
        """Validate agent parameters."""
        if not self.skills:
            raise ValueError("Skills cannot be empty")
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 <= self.availability <= 1):
            raise ValueError("Availability must be between 0 and 1")
    
    def __hash__(self):
        """Make agent hashable for use in sets/dicts."""
        return hash((self.agent_id, tuple(self.skills), self.capacity))


@dataclass(frozen=True)
class Task:
    """Represents a task to be scheduled."""
    
    task_id: str
    required_skills: List[str]
    priority: int
    duration: int
    dependencies: List[str] = field(default_factory=list)
    
    def __init__(self, id: str = None, task_id: str = None, required_skills: List[str] = None,
                 priority: int = 1, duration: int = 1, dependencies: List[str] = None):
        """Initialize task with flexible parameter names."""
        # Handle both 'id' and 'task_id' parameter names
        if id is not None and task_id is None:
            task_id = id
        elif task_id is None and id is None:
            raise ValueError("Either 'id' or 'task_id' must be provided")
            
        object.__setattr__(self, 'task_id', task_id)
        object.__setattr__(self, 'required_skills', required_skills or [])
        object.__setattr__(self, 'priority', priority)
        object.__setattr__(self, 'duration', duration)
        object.__setattr__(self, 'dependencies', dependencies or [])
        
        self.__post_init__()
    
    @property
    def id(self) -> str:
        """Alias for task_id for API compatibility."""
        return self.task_id
    
    def __post_init__(self):
        """Validate task parameters."""
        if not self.required_skills:
            raise ValueError("Required skills cannot be empty")
        if self.priority <= 0:
            raise ValueError("Priority must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
    
    def can_be_assigned_to(self, agent: Agent) -> bool:
        """Check if this task can be assigned to the given agent."""
        agent_skills = set(agent.skills)
        required_skills = set(self.required_skills)
        return required_skills.issubset(agent_skills)


@dataclass(frozen=True)
class TimeWindowTask(Task):
    """Represents a task with time window constraints."""
    
    earliest_start: int = 0
    latest_finish: int = float('inf')
    
    def __init__(self, id: str = None, task_id: str = None, required_skills: List[str] = None,
                 priority: int = 1, duration: int = 1, dependencies: List[str] = None,
                 earliest_start: int = 0, latest_finish: int = float('inf')):
        """Initialize time window task with flexible parameter names."""
        # Handle both 'id' and 'task_id' parameter names
        if id is not None and task_id is None:
            task_id = id
        elif task_id is None and id is None:
            raise ValueError("Either 'id' or 'task_id' must be provided")
            
        object.__setattr__(self, 'task_id', task_id)
        object.__setattr__(self, 'required_skills', required_skills or [])
        object.__setattr__(self, 'priority', priority)
        object.__setattr__(self, 'duration', duration)
        object.__setattr__(self, 'dependencies', dependencies or [])
        object.__setattr__(self, 'earliest_start', earliest_start)
        object.__setattr__(self, 'latest_finish', latest_finish)
        
        self.__post_init__()
    
    def __post_init__(self):
        """Validate time window task parameters."""
        super().__post_init__()
        
        if self.earliest_start < 0:
            raise ValueError("Earliest start must be non-negative")
        
        if self.earliest_start + self.duration > self.latest_finish:
            raise ValueError("Task cannot be completed within time window")
    
    def is_feasible_at_time(self, start_time: int) -> bool:
        """Check if task can start at given time and finish within window."""
        if start_time < self.earliest_start:
            return False
        if start_time + self.duration > self.latest_finish:
            return False
        return True


@dataclass
class Solution:
    """Represents a solution to the task scheduling problem."""
    
    assignments: Dict[str, str]  # task_id -> agent_id
    makespan: float
    cost: float
    backend_used: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    schedule: Optional[Dict[str, Dict[str, tuple]]] = None
    task_durations: Optional[Dict[str, int]] = None
    total_cost: Optional[float] = None
    
    def __post_init__(self):
        """Validate solution parameters."""
        if not self.assignments:
            raise ValueError("Assignments cannot be empty")
        if self.makespan < 0:
            raise ValueError("Makespan must be non-negative")
    
    def get_load_distribution(self) -> Dict[str, int]:
        """Get the load distribution across agents."""
        return dict(Counter(self.assignments.values()))
    
    def get_assigned_agents(self) -> Set[str]:
        """Get the set of agents that have been assigned tasks."""
        return set(self.assignments.values())
    
    def get_task_count(self) -> int:
        """Get the total number of tasks in this solution."""
        return len(self.assignments)
    
    def calculate_quality_score(
        self, 
        weight_makespan: float = 0.4,
        weight_cost: float = 0.3,
        weight_balance: float = 0.3
    ) -> float:
        """Calculate a quality score for this solution."""
        # Normalize makespan (lower is better)
        makespan_score = max(0, 1 - (self.makespan / 100))  # Assume max makespan of 100
        
        # Normalize cost (lower is better)
        cost_score = max(0, 1 - (self.cost / 50))  # Assume max cost of 50
        
        # Calculate load balance (more balanced is better)
        load_dist = list(self.get_load_distribution().values())
        if len(load_dist) > 1:
            avg_load = sum(load_dist) / len(load_dist)
            variance = sum((load - avg_load) ** 2 for load in load_dist) / len(load_dist)
            balance_score = max(0, 1 - (variance / avg_load))
        else:
            balance_score = 1.0
        
        # Weighted combination
        quality = (
            weight_makespan * makespan_score +
            weight_cost * cost_score +
            weight_balance * balance_score
        )
        
        return min(1.0, max(0.0, quality))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary for serialization."""
        return {
            "assignments": self.assignments,
            "makespan": self.makespan,
            "cost": self.cost,
            "backend_used": self.backend_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solution":
        """Create solution from dictionary."""
        return cls(
            assignments=data["assignments"],
            makespan=data["makespan"],
            cost=data["cost"],
            backend_used=data["backend_used"]
        )