"""Comprehensive validation module for quantum task planner."""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import Agent, Task, TimeWindowTask, Solution

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    results: List[ValidationResult]
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error-level validation results."""
        return [r for r in self.results if r.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get all warning-level validation results."""
        return [r for r in self.results if r.severity == ValidationSeverity.WARNING]
    
    @property
    def critical(self) -> List[ValidationResult]:
        """Get all critical-level validation results."""
        return [r for r in self.results if r.severity == ValidationSeverity.CRITICAL]


class InputValidator:
    """Comprehensive input validation for quantum task planner."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
    
    def validate_agents(self, agents: List[Agent]) -> ValidationReport:
        """Validate agent inputs comprehensively."""
        results = []
        
        # Basic existence check
        if not agents:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="No agents provided",
                suggestion="Provide at least one agent"
            ))
            return ValidationReport(is_valid=False, results=results)
        
        # Check for duplicates
        agent_ids = [agent.agent_id for agent in agents]
        duplicates = set([id for id in agent_ids if agent_ids.count(id) > 1])
        if duplicates:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duplicate agent IDs found: {duplicates}",
                suggestion="Ensure all agent IDs are unique"
            ))
        
        # Validate individual agents
        for i, agent in enumerate(agents):
            agent_results = self._validate_single_agent(agent, i)
            results.extend(agent_results)
        
        # Check skill distribution
        all_skills = set()
        for agent in agents:
            all_skills.update(agent.skills)
        
        if len(all_skills) < 2:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Limited skill diversity: only {len(all_skills)} unique skills",
                suggestion="Consider adding agents with diverse skills for better optimization"
            ))
        
        # Check capacity distribution
        capacities = [agent.capacity for agent in agents]
        if len(set(capacities)) == 1:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="All agents have identical capacity",
                suggestion="Different capacities can improve optimization"
            ))
        
        is_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for r in results)
        if self.strict_mode:
            is_valid = is_valid and not any(r.severity == ValidationSeverity.WARNING for r in results)
        
        return ValidationReport(is_valid=is_valid, results=results)
    
    def _validate_single_agent(self, agent: Agent, index: int) -> List[ValidationResult]:
        """Validate a single agent."""
        results = []
        
        # Agent ID validation
        if not agent.agent_id or not agent.agent_id.strip():
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Agent {index} has empty or invalid ID",
                field="agent_id",
                suggestion="Provide a non-empty string ID"
            ))
        
        # Skills validation
        if not agent.skills:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Agent {agent.agent_id} has no skills defined",
                field="skills",
                suggestion="Add at least one skill to the agent"
            ))
        else:
            # Check for empty skills
            empty_skills = [skill for skill in agent.skills if not skill or not skill.strip()]
            if empty_skills:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Agent {agent.agent_id} has empty skill entries",
                    field="skills",
                    suggestion="Remove empty skill entries"
                ))
            
            # Check for duplicate skills
            if len(agent.skills) != len(set(agent.skills)):
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Agent {agent.agent_id} has duplicate skills",
                    field="skills",
                    suggestion="Remove duplicate skill entries"
                ))
        
        # Capacity validation
        if agent.capacity <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Agent {agent.agent_id} has invalid capacity: {agent.capacity}",
                field="capacity",
                suggestion="Set capacity to a positive number"
            ))
        elif agent.capacity > 100:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Agent {agent.agent_id} has very high capacity: {agent.capacity}",
                field="capacity",
                suggestion="Verify this capacity is realistic"
            ))
        
        return results
    
    def validate_tasks(self, tasks: List[Task]) -> ValidationReport:
        """Validate task inputs comprehensively."""
        results = []
        
        # Basic existence check
        if not tasks:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="No tasks provided",
                suggestion="Provide at least one task"
            ))
            return ValidationReport(is_valid=False, results=results)
        
        # Check for duplicates
        task_ids = [task.task_id for task in tasks]
        duplicates = set([id for id in task_ids if task_ids.count(id) > 1])
        if duplicates:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Duplicate task IDs found: {duplicates}",
                suggestion="Ensure all task IDs are unique"
            ))
        
        # Validate individual tasks
        for i, task in enumerate(tasks):
            task_results = self._validate_single_task(task, i)
            results.extend(task_results)
        
        # Check priority distribution
        priorities = [task.priority for task in tasks]
        if len(set(priorities)) == 1:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="All tasks have identical priority",
                suggestion="Different priorities can improve optimization"
            ))
        
        # Check duration distribution
        durations = [task.duration for task in tasks]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations)
        
        if avg_duration > 50:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"High average task duration: {avg_duration:.1f}",
                suggestion="Consider breaking down large tasks"
            ))
        
        is_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for r in results)
        if self.strict_mode:
            is_valid = is_valid and not any(r.severity == ValidationSeverity.WARNING for r in results)
        
        return ValidationReport(is_valid=is_valid, results=results)
    
    def _validate_single_task(self, task: Task, index: int) -> List[ValidationResult]:
        """Validate a single task."""
        results = []
        
        # Task ID validation
        if not task.task_id or not task.task_id.strip():
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task {index} has empty or invalid ID",
                field="task_id",
                suggestion="Provide a non-empty string ID"
            ))
        
        # Required skills validation
        if not task.required_skills:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task {task.task_id} has no required skills",
                field="required_skills",
                suggestion="Add at least one required skill"
            ))
        else:
            # Check for empty skills
            empty_skills = [skill for skill in task.required_skills if not skill or not skill.strip()]
            if empty_skills:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Task {task.task_id} has empty required skill entries",
                    field="required_skills",
                    suggestion="Remove empty skill entries"
                ))
        
        # Priority validation
        if task.priority <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task {task.task_id} has invalid priority: {task.priority}",
                field="priority",
                suggestion="Set priority to a positive number"
            ))
        elif task.priority > 100:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Task {task.task_id} has very high priority: {task.priority}",
                field="priority",
                suggestion="Verify this priority is realistic"
            ))
        
        # Duration validation
        if task.duration <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Task {task.task_id} has invalid duration: {task.duration}",
                field="duration",
                suggestion="Set duration to a positive number"
            ))
        elif task.duration > 1000:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Task {task.task_id} has very long duration: {task.duration}",
                field="duration",
                suggestion="Consider breaking down this large task"
            ))
        
        # Time window validation for TimeWindowTask
        if isinstance(task, TimeWindowTask):
            if task.earliest_start < 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Task {task.task_id} has negative earliest_start: {task.earliest_start}",
                    field="earliest_start",
                    suggestion="Set earliest_start to a non-negative number"
                ))
            
            if task.latest_finish <= task.earliest_start:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Task {task.task_id} has latest_finish <= earliest_start",
                    field="latest_finish",
                    suggestion="Ensure latest_finish > earliest_start"
                ))
            
            if task.latest_finish < task.earliest_start + task.duration:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Task {task.task_id} time window too narrow for duration",
                    field="time_window",
                    suggestion="Increase latest_finish or reduce duration"
                ))
        
        return results
    
    def validate_compatibility(self, agents: List[Agent], tasks: List[Task]) -> ValidationReport:
        """Validate agent-task compatibility."""
        results = []
        
        # Get all available skills
        all_agent_skills = set()
        for agent in agents:
            all_agent_skills.update(agent.skills)
        
        # Get all required skills
        all_required_skills = set()
        for task in tasks:
            all_required_skills.update(task.required_skills)
        
        # Check for skills that no agent has
        missing_skills = all_required_skills - all_agent_skills
        if missing_skills:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"No agents have required skills: {missing_skills}",
                suggestion="Add agents with these skills or modify task requirements"
            ))
        
        # Check assignability
        assignable_tasks = []
        unassignable_tasks = []
        
        for task in tasks:
            can_assign = any(task.can_be_assigned_to(agent) for agent in agents)
            if can_assign:
                assignable_tasks.append(task.task_id)
            else:
                unassignable_tasks.append(task.task_id)
        
        if unassignable_tasks:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Tasks cannot be assigned: {unassignable_tasks}",
                suggestion="Check skill requirements and agent capabilities"
            ))
        
        # Check capacity vs task count
        total_capacity = sum(agent.capacity for agent in agents)
        task_count = len(tasks)
        
        if total_capacity < task_count:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Total capacity ({total_capacity}) < task count ({task_count})",
                suggestion="Some agents may be overloaded"
            ))
        
        # Check total workload
        total_duration = sum(task.duration for task in tasks)
        if total_capacity > 0:
            avg_load = total_duration / total_capacity
            if avg_load > 10:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"High workload per capacity unit: {avg_load:.1f}",
                    suggestion="Consider adding more agents or reducing task durations"
                ))
        
        is_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for r in results)
        if self.strict_mode:
            is_valid = is_valid and not any(r.severity == ValidationSeverity.WARNING for r in results)
        
        return ValidationReport(is_valid=is_valid, results=results)
    
    def validate_constraints(self, constraints: Dict[str, Any]) -> ValidationReport:
        """Validate constraint dictionary."""
        results = []
        
        if not isinstance(constraints, dict):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Constraints must be a dictionary",
                suggestion="Provide constraints as a dict"
            ))
            return ValidationReport(is_valid=False, results=results)
        
        # Known constraint types
        known_constraints = {
            "skill_match", "capacity_limit", "precedence", "time_windows", 
            "time_horizon", "resource_limits", "deadline"
        }
        
        # Check for unknown constraints
        unknown = set(constraints.keys()) - known_constraints
        if unknown:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Unknown constraints: {unknown}",
                suggestion=f"Known constraints: {known_constraints}"
            ))
        
        # Validate precedence constraints
        if "precedence" in constraints:
            precedence = constraints["precedence"]
            if not isinstance(precedence, dict):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Precedence constraint must be a dictionary",
                    field="precedence",
                    suggestion="Use format: {'task_id': ['prerequisite1', 'prerequisite2']}"
                ))
        
        # Validate time horizon
        if "time_horizon" in constraints:
            horizon = constraints["time_horizon"]
            if not isinstance(horizon, (int, float)) or horizon <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid time_horizon: {horizon}",
                    field="time_horizon",
                    suggestion="Set time_horizon to a positive number"
                ))
        
        is_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for r in results)
        if self.strict_mode:
            is_valid = is_valid and not any(r.severity == ValidationSeverity.WARNING for r in results)
        
        return ValidationReport(is_valid=is_valid, results=results)
    
    def validate_solution(self, solution: Solution, agents: List[Agent], tasks: List[Task]) -> ValidationReport:
        """Validate a solution comprehensively."""
        results = []
        
        # Basic solution structure
        if not solution.assignments:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Solution has no assignments",
                suggestion="Ensure optimization produced valid assignments"
            ))
            return ValidationReport(is_valid=False, results=results)
        
        # Check all tasks are assigned
        task_ids = {task.task_id for task in tasks}
        assigned_tasks = set(solution.assignments.keys())
        missing_tasks = task_ids - assigned_tasks
        
        if missing_tasks:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Tasks not assigned: {missing_tasks}",
                suggestion="Ensure all tasks are included in the solution"
            ))
        
        # Check for invalid agent assignments
        agent_ids = {agent.agent_id for agent in agents}
        assigned_agents = set(solution.assignments.values())
        invalid_agents = assigned_agents - agent_ids
        
        if invalid_agents:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Invalid agent assignments: {invalid_agents}",
                suggestion="Verify agent IDs in solution"
            ))
        
        # Check skill compatibility
        skill_violations = []
        for task_id, agent_id in solution.assignments.items():
            task = next((t for t in tasks if t.task_id == task_id), None)
            agent = next((a for a in agents if a.agent_id == agent_id), None)
            
            if task and agent and not task.can_be_assigned_to(agent):
                skill_violations.append(f"{task_id} -> {agent_id}")
        
        if skill_violations:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Skill violations: {skill_violations}",
                suggestion="Check skill requirements and agent capabilities"
            ))
        
        # Check capacity violations
        agent_loads = {}
        for task_id, agent_id in solution.assignments.items():
            task = next((t for t in tasks if t.task_id == task_id), None)
            if task:
                agent_loads[agent_id] = agent_loads.get(agent_id, 0) + 1
        
        capacity_violations = []
        for agent in agents:
            load = agent_loads.get(agent.agent_id, 0)
            if load > agent.capacity:
                capacity_violations.append(f"{agent.agent_id}: {load}/{agent.capacity}")
        
        if capacity_violations:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Capacity exceeded: {capacity_violations}",
                suggestion="Consider load balancing or adding capacity"
            ))
        
        # Validate solution metrics
        if solution.makespan < 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid makespan: {solution.makespan}",
                field="makespan",
                suggestion="Makespan should be non-negative"
            ))
        
        if solution.cost < 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid cost: {solution.cost}",
                field="cost",
                suggestion="Cost should be non-negative"
            ))
        
        is_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for r in results)
        if self.strict_mode:
            is_valid = is_valid and not any(r.severity == ValidationSeverity.WARNING for r in results)
        
        return ValidationReport(is_valid=is_valid, results=results)
    
    def validate_all(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Optional[Dict[str, Any]] = None,
        solution: Optional[Solution] = None
    ) -> ValidationReport:
        """Perform comprehensive validation of all inputs."""
        all_results = []
        
        # Validate agents
        agent_report = self.validate_agents(agents)
        all_results.extend(agent_report.results)
        
        # Validate tasks
        task_report = self.validate_tasks(tasks)
        all_results.extend(task_report.results)
        
        # Validate compatibility (only if agents and tasks are individually valid)
        if agent_report.is_valid and task_report.is_valid:
            compat_report = self.validate_compatibility(agents, tasks)
            all_results.extend(compat_report.results)
        
        # Validate constraints
        if constraints:
            constraint_report = self.validate_constraints(constraints)
            all_results.extend(constraint_report.results)
        
        # Validate solution
        if solution:
            solution_report = self.validate_solution(solution, agents, tasks)
            all_results.extend(solution_report.results)
        
        is_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for r in all_results)
        if self.strict_mode:
            is_valid = is_valid and not any(r.severity == ValidationSeverity.WARNING for r in all_results)
        
        return ValidationReport(is_valid=is_valid, results=all_results)


def format_validation_report(report: ValidationReport) -> str:
    """Format validation report for display."""
    lines = []
    lines.append(f"Validation {'PASSED' if report.is_valid else 'FAILED'}")
    lines.append(f"{'='*50}")
    
    if report.critical:
        lines.append("\n🔴 CRITICAL ISSUES:")
        for result in report.critical:
            lines.append(f"  • {result.message}")
            if result.suggestion:
                lines.append(f"    💡 {result.suggestion}")
    
    if report.errors:
        lines.append("\n❌ ERRORS:")
        for result in report.errors:
            lines.append(f"  • {result.message}")
            if result.suggestion:
                lines.append(f"    💡 {result.suggestion}")
    
    if report.warnings:
        lines.append("\n⚠️  WARNINGS:")
        for result in report.warnings:
            lines.append(f"  • {result.message}")
            if result.suggestion:
                lines.append(f"    💡 {result.suggestion}")
    
    info_results = [r for r in report.results if r.severity == ValidationSeverity.INFO]
    if info_results:
        lines.append("\nℹ️  INFO:")
        for result in info_results:
            lines.append(f"  • {result.message}")
    
    return "\n".join(lines)