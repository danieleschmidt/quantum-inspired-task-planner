"""QUBO (Quadratic Unconstrained Binary Optimization) formulation engine."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

from quantum_planner.models import Agent, Task, TimeWindowTask

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Available objective functions."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MAXIMIZE_PRIORITY = "maximize_priority"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_COST = "minimize_cost"


@dataclass
class QUBOConstraint:
    """Represents a constraint in QUBO formulation."""
    name: str
    penalty: float
    constraint_type: str
    parameters: Dict[str, Any]


class QUBOBuilder:
    """Builds QUBO matrices for task assignment optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QUBOBuilder")
        self.constraints: List[QUBOConstraint] = []
        self.objective: Optional[ObjectiveType] = None
        self.variable_map: Dict[Tuple[str, str], int] = {}
        self.num_variables = 0
    
    def add_objective(
        self, 
        objective_type: ObjectiveType, 
        weight: float = 1.0
    ) -> None:
        """Add primary objective function."""
        self.objective = objective_type
        self.objective_weight = weight
        self.logger.debug(f"Added objective: {objective_type.value} with weight {weight}")
    
    def add_constraint(
        self,
        name: str,
        constraint_type: str,
        penalty: float,
        **parameters
    ) -> None:
        """Add constraint to QUBO formulation."""
        constraint = QUBOConstraint(
            name=name,
            penalty=penalty,
            constraint_type=constraint_type,
            parameters=parameters
        )
        self.constraints.append(constraint)
        self.logger.debug(f"Added constraint: {name} with penalty {penalty}")
    
    def build_qubo(
        self,
        agents: List[Agent],
        tasks: List[Task]
    ) -> Tuple[np.ndarray, Dict[int, Tuple[str, str]]]:
        """Build QUBO matrix for task assignment problem."""
        self.logger.info(f"Building QUBO for {len(tasks)} tasks and {len(agents)} agents")
        
        # Create variable mapping: x[task_id, agent_id] -> variable_index
        self._create_variable_mapping(agents, tasks)
        
        # Initialize QUBO matrix
        Q = np.zeros((self.num_variables, self.num_variables))
        
        # Add objective function
        if self.objective:
            self._add_objective_terms(Q, agents, tasks)
        
        # Add constraints
        for constraint in self.constraints:
            self._add_constraint_terms(Q, constraint, agents, tasks)
        
        # Create reverse mapping for solution parsing
        reverse_map = {v: k for k, v in self.variable_map.items()}
        
        self.logger.info(f"QUBO matrix built: {Q.shape[0]} variables")
        return Q, reverse_map
    
    def _create_variable_mapping(self, agents: List[Agent], tasks: List[Task]) -> None:
        """Create mapping from (task_id, agent_id) to variable indices."""
        self.variable_map.clear()
        var_idx = 0
        
        for task in tasks:
            for agent in agents:
                # Only create variables for feasible assignments
                if task.can_be_assigned_to(agent):
                    self.variable_map[(task.task_id, agent.agent_id)] = var_idx
                    var_idx += 1
        
        self.num_variables = var_idx
        self.logger.debug(f"Created {self.num_variables} binary variables")
    
    def _add_objective_terms(
        self,
        Q: np.ndarray,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Add objective function terms to QUBO matrix."""
        if self.objective == ObjectiveType.MINIMIZE_MAKESPAN:
            self._add_makespan_objective(Q, agents, tasks)
        elif self.objective == ObjectiveType.MAXIMIZE_PRIORITY:
            self._add_priority_objective(Q, agents, tasks)
        elif self.objective == ObjectiveType.BALANCE_LOAD:
            self._add_load_balance_objective(Q, agents, tasks)
        elif self.objective == ObjectiveType.MINIMIZE_COST:
            self._add_cost_objective(Q, agents, tasks)
    
    def _add_makespan_objective(
        self,
        Q: np.ndarray,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Add makespan minimization terms."""
        for task in tasks:
            for agent in agents:
                if (task.task_id, agent.agent_id) in self.variable_map:
                    var_idx = self.variable_map[(task.task_id, agent.agent_id)]
                    # Normalized task duration divided by agent capacity
                    cost = (task.duration / agent.capacity) * self.objective_weight
                    Q[var_idx, var_idx] += cost
    
    def _add_priority_objective(
        self,
        Q: np.ndarray,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Add priority maximization terms (negative for minimization)."""
        max_priority = max(task.priority for task in tasks)
        
        for task in tasks:
            for agent in agents:
                if (task.task_id, agent.agent_id) in self.variable_map:
                    var_idx = self.variable_map[(task.task_id, agent.agent_id)]
                    # Negative priority for maximization in minimization framework
                    priority_cost = -(task.priority / max_priority) * self.objective_weight
                    Q[var_idx, var_idx] += priority_cost
    
    def _add_load_balance_objective(
        self,
        Q: np.ndarray,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Add load balancing terms."""
        # Quadratic penalty for load imbalance
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i != j:
                    for agent in agents:
                        if ((task1.task_id, agent.agent_id) in self.variable_map and
                            (task2.task_id, agent.agent_id) in self.variable_map):
                            
                            var1 = self.variable_map[(task1.task_id, agent.agent_id)]
                            var2 = self.variable_map[(task2.task_id, agent.agent_id)]
                            
                            # Encourage load balance
                            balance_penalty = (task1.duration * task2.duration) / (agent.capacity ** 2)
                            balance_penalty *= self.objective_weight
                            Q[var1, var2] += balance_penalty
    
    def _add_cost_objective(
        self,
        Q: np.ndarray,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Add cost minimization terms."""
        for task in tasks:
            for agent in agents:
                if (task.task_id, agent.agent_id) in self.variable_map:
                    var_idx = self.variable_map[(task.task_id, agent.agent_id)]
                    # Simple cost model based on task complexity and agent efficiency
                    cost = task.duration * (1.0 / agent.capacity) * self.objective_weight
                    Q[var_idx, var_idx] += cost
    
    def _add_constraint_terms(
        self,
        Q: np.ndarray,
        constraint: QUBOConstraint,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Add constraint penalty terms to QUBO matrix."""
        if constraint.constraint_type == "one_task_one_agent":
            self._add_assignment_constraint(Q, constraint.penalty, agents, tasks)
        elif constraint.constraint_type == "capacity_limit":
            self._add_capacity_constraint(Q, constraint.penalty, agents, tasks)
        elif constraint.constraint_type == "skill_matching":
            self._add_skill_constraint(Q, constraint.penalty, agents, tasks)
        elif constraint.constraint_type == "precedence":
            self._add_precedence_constraint(Q, constraint.penalty, agents, tasks, constraint.parameters)
        elif constraint.constraint_type == "time_window":
            self._add_time_window_constraint(Q, constraint.penalty, agents, tasks, constraint.parameters)
    
    def _add_assignment_constraint(
        self,
        Q: np.ndarray,
        penalty: float,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Each task assigned to exactly one agent."""
        for task in tasks:
            # Get all variables for this task
            task_vars = []
            for agent in agents:
                if (task.task_id, agent.agent_id) in self.variable_map:
                    task_vars.append(self.variable_map[(task.task_id, agent.agent_id)])
            
            # Add penalty for not assigning exactly one agent
            # (x1 + x2 + ... + xn - 1)^2 = sum(xi^2) + 2*sum(xi*xj) - 2*sum(xi) + 1
            
            # Linear terms: -2*sum(xi)
            for var in task_vars:
                Q[var, var] -= 2 * penalty
            
            # Quadratic terms: 2*sum(xi*xj)
            for i, var1 in enumerate(task_vars):
                for j, var2 in enumerate(task_vars):
                    if i < j:  # Avoid double counting
                        Q[var1, var2] += 2 * penalty
            
            # Constant term (+1) is ignored in QUBO optimization
    
    def _add_capacity_constraint(
        self,
        Q: np.ndarray,
        penalty: float,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Agent capacity constraints."""
        for agent in agents:
            # Get all tasks that could be assigned to this agent
            agent_vars = []
            task_durations = []
            
            for task in tasks:
                if (task.task_id, agent.agent_id) in self.variable_map:
                    agent_vars.append(self.variable_map[(task.task_id, agent.agent_id)])
                    task_durations.append(task.duration)
            
            # Penalty for exceeding capacity
            # (sum(duration_i * x_i) - capacity)^2 if sum > capacity
            # Approximated as quadratic terms
            
            for i, var1 in enumerate(agent_vars):
                for j, var2 in enumerate(agent_vars):
                    if i <= j:
                        duration_product = task_durations[i] * task_durations[j]
                        if duration_product > agent.capacity:
                            capacity_penalty = (duration_product / agent.capacity) * penalty
                            if i == j:
                                Q[var1, var1] += capacity_penalty
                            else:
                                Q[var1, var2] += capacity_penalty
    
    def _add_skill_constraint(
        self,
        Q: np.ndarray,
        penalty: float,
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Skill matching constraints (soft constraint for preference)."""
        for task in tasks:
            for agent in agents:
                if (task.task_id, agent.agent_id) in self.variable_map:
                    var_idx = self.variable_map[(task.task_id, agent.agent_id)]
                    
                    # Calculate skill match quality
                    task_skills = set(task.required_skills)
                    agent_skills = set(agent.skills)
                    
                    if not task_skills.issubset(agent_skills):
                        # Penalty for insufficient skills
                        missing_skills = len(task_skills - agent_skills)
                        skill_penalty = missing_skills * penalty
                        Q[var_idx, var_idx] += skill_penalty
    
    def _add_precedence_constraint(
        self,
        Q: np.ndarray,
        penalty: float,
        agents: List[Agent],
        tasks: List[Task],
        parameters: Dict[str, Any]
    ) -> None:
        """Task precedence constraints."""
        precedences = parameters.get("precedences", {})
        
        for successor_id, predecessors in precedences.items():
            # Find successor task
            successor_task = None
            for task in tasks:
                if task.task_id == successor_id:
                    successor_task = task
                    break
            
            if not successor_task:
                continue
            
            for predecessor_id in predecessors:
                # Find predecessor task
                predecessor_task = None
                for task in tasks:
                    if task.task_id == predecessor_id:
                        predecessor_task = task
                        break
                
                if not predecessor_task:
                    continue
                
                # Add penalty for violating precedence
                for agent in agents:
                    if ((predecessor_task.task_id, agent.agent_id) in self.variable_map and
                        (successor_task.task_id, agent.agent_id) in self.variable_map):
                        
                        pred_var = self.variable_map[(predecessor_task.task_id, agent.agent_id)]
                        succ_var = self.variable_map[(successor_task.task_id, agent.agent_id)]
                        
                        # Penalty if successor assigned but predecessor not
                        Q[succ_var, succ_var] += penalty
                        Q[pred_var, succ_var] -= penalty
    
    def _add_time_window_constraint(
        self,
        Q: np.ndarray,
        penalty: float,
        agents: List[Agent],
        tasks: List[Task],
        parameters: Dict[str, Any]
    ) -> None:
        """Time window constraints for TimeWindowTask."""
        time_horizon = parameters.get("time_horizon", 24)
        
        for task in tasks:
            if isinstance(task, TimeWindowTask):
                for agent in agents:
                    if (task.task_id, agent.agent_id) in self.variable_map:
                        var_idx = self.variable_map[(task.task_id, agent.agent_id)]
                        
                        # Check if assignment is feasible within time window
                        if not task.is_feasible_at_time(task.earliest_start):
                            # High penalty for infeasible time window assignments
                            Q[var_idx, var_idx] += penalty * 10
    
    def validate_qubo(self, Q: np.ndarray) -> bool:
        """Validate QUBO matrix properties."""
        if Q.shape[0] != Q.shape[1]:
            return False
        
        # Check if matrix is symmetric (QUBO should be upper triangular)
        if not np.allclose(Q, Q.T):
            self.logger.warning("QUBO matrix is not symmetric")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
            return False
        
        return True
    
    def get_problem_stats(self) -> Dict[str, Any]:
        """Get statistics about the formulated problem."""
        return {
            "num_variables": self.num_variables,
            "num_constraints": len(self.constraints),
            "objective": self.objective.value if self.objective else None,
            "variable_density": len(self.variable_map) / (len(self.constraints) * 10) if self.constraints else 0
        }


def create_standard_qubo(
    agents: List[Agent],
    tasks: List[Task],
    objective: str = "minimize_makespan",
    constraint_penalties: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, Dict[int, Tuple[str, str]]]:
    """Create a standard QUBO formulation with common constraints."""
    
    if constraint_penalties is None:
        constraint_penalties = {
            "assignment": 1000.0,
            "capacity": 100.0,
            "skill": 50.0
        }
    
    builder = QUBOBuilder()
    
    # Add objective
    obj_type = ObjectiveType(objective)
    builder.add_objective(obj_type, weight=1.0)
    
    # Add standard constraints
    builder.add_constraint(
        "assignment",
        "one_task_one_agent",
        constraint_penalties["assignment"]
    )
    
    builder.add_constraint(
        "capacity",
        "capacity_limit", 
        constraint_penalties["capacity"]
    )
    
    builder.add_constraint(
        "skill_match",
        "skill_matching",
        constraint_penalties["skill"]
    )
    
    return builder.build_qubo(agents, tasks)