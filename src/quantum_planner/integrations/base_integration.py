"""Base integration framework for agent orchestration platforms."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging

from quantum_planner.models import Agent, Task, Solution
from quantum_planner.optimizer import optimize_tasks

logger = logging.getLogger(__name__)


class IntegrationError(Exception):
    """Base exception for integration errors."""
    pass


@dataclass
class IntegrationConfig:
    """Configuration for framework integrations."""
    backend: str = "simulator"
    optimization_objective: str = "minimize_makespan"
    enable_real_time: bool = False
    update_interval: float = 30.0
    fallback_enabled: bool = True
    logging_enabled: bool = True


class BaseIntegration(ABC):
    """Base class for agent framework integrations."""
    
    def __init__(
        self,
        framework_name: str,
        config: Optional[IntegrationConfig] = None
    ):
        self.framework_name = framework_name
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger(f"{__name__}.{framework_name}")
        
        self._agent_mapping: Dict[str, Any] = {}
        self._task_mapping: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            'assignment_complete': [],
            'task_started': [],
            'task_completed': [],
            'error_occurred': []
        }
    
    @abstractmethod
    def extract_agents(self, framework_agents: List[Any]) -> List[Agent]:
        """Extract quantum planner agents from framework-specific agents."""
        pass
    
    @abstractmethod
    def extract_tasks(self, framework_tasks: List[Any]) -> List[Task]:
        """Extract quantum planner tasks from framework-specific tasks."""
        pass
    
    @abstractmethod
    def apply_assignments(
        self,
        solution: Solution,
        framework_agents: List[Any],
        framework_tasks: List[Any]
    ) -> Dict[str, Any]:
        """Apply quantum optimization results to framework."""
        pass
    
    def optimize_framework_tasks(
        self,
        framework_agents: List[Any],
        framework_tasks: List[Any],
        **optimization_params
    ) -> Dict[str, Any]:
        """Main optimization workflow for framework integration."""
        
        try:
            self.logger.info(f"Starting {self.framework_name} optimization")
            
            # Extract quantum planner compatible objects
            agents = self.extract_agents(framework_agents)
            tasks = self.extract_tasks(framework_tasks)
            
            self.logger.debug(f"Extracted {len(agents)} agents and {len(tasks)} tasks")
            
            # Validate inputs
            if not agents:
                raise IntegrationError("No valid agents found")
            if not tasks:
                raise IntegrationError("No valid tasks found")
            
            # Run quantum optimization
            solution = optimize_tasks(
                agents=agents,
                tasks=tasks,
                backend=self.config.backend,
                **optimization_params
            )
            
            self.logger.info(f"Optimization completed: makespan={solution.makespan:.2f}")
            
            # Apply results to framework
            assignment_result = self.apply_assignments(
                solution, framework_agents, framework_tasks
            )
            
            # Trigger callbacks
            self._trigger_callback('assignment_complete', solution, assignment_result)
            
            return {
                'success': True,
                'solution': solution,
                'assignment_result': assignment_result,
                'framework': self.framework_name
            }
            
        except Exception as e:
            error_msg = f"{self.framework_name} optimization failed: {str(e)}"
            self.logger.error(error_msg)
            self._trigger_callback('error_occurred', e)
            
            if self.config.fallback_enabled:
                return self._fallback_assignment(framework_agents, framework_tasks)
            else:
                raise IntegrationError(error_msg) from e
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for integration events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def _trigger_callback(self, event: str, *args) -> None:
        """Trigger registered callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")
    
    def _fallback_assignment(
        self,
        framework_agents: List[Any],
        framework_tasks: List[Any]
    ) -> Dict[str, Any]:
        """Fallback assignment strategy when optimization fails."""
        
        self.logger.info("Using fallback assignment strategy")
        
        try:
            # Simple round-robin assignment
            assignments = {}
            
            for i, task in enumerate(framework_tasks):
                agent_index = i % len(framework_agents)
                agent = framework_agents[agent_index]
                
                # Create mapping
                task_id = self._get_task_id(task)
                agent_id = self._get_agent_id(agent)
                assignments[task_id] = agent_id
            
            return {
                'success': True,
                'solution': None,
                'assignment_result': {
                    'assignments': assignments,
                    'method': 'fallback_round_robin'
                },
                'framework': self.framework_name
            }
            
        except Exception as e:
            self.logger.error(f"Fallback assignment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'framework': self.framework_name
            }
    
    def _get_agent_id(self, framework_agent: Any) -> str:
        """Extract agent ID from framework-specific agent object."""
        
        # Try common attribute names
        for attr in ['id', 'agent_id', 'name', 'role']:
            if hasattr(framework_agent, attr):
                return str(getattr(framework_agent, attr))
        
        # Fallback to string representation
        return str(framework_agent)
    
    def _get_task_id(self, framework_task: Any) -> str:
        """Extract task ID from framework-specific task object."""
        
        # Try common attribute names
        for attr in ['id', 'task_id', 'name', 'description']:
            if hasattr(framework_task, attr):
                value = getattr(framework_task, attr)
                if value:
                    return str(value)[:50]  # Limit length
        
        # Fallback to string representation
        return str(framework_task)[:50]
    
    def _extract_skills(self, framework_agent: Any) -> List[str]:
        """Extract skills from framework-specific agent object."""
        
        skills = []
        
        # Try common skill attributes
        for attr in ['skills', 'capabilities', 'tools', 'role']:
            if hasattr(framework_agent, attr):
                value = getattr(framework_agent, attr)
                
                if isinstance(value, list):
                    skills.extend([str(s) for s in value])
                elif isinstance(value, str):
                    skills.append(value)
                elif hasattr(value, '__iter__'):  # Iterable
                    skills.extend([str(s) for s in value])
        
        # Ensure at least one skill
        if not skills:
            skills = [self._get_agent_id(framework_agent)]
        
        return list(set(skills))  # Remove duplicates
    
    def _extract_task_requirements(self, framework_task: Any) -> List[str]:
        """Extract required skills from framework-specific task object."""
        
        requirements = []
        
        # Try common requirement attributes
        for attr in ['required_skills', 'requirements', 'tools', 'dependencies']:
            if hasattr(framework_task, attr):
                value = getattr(framework_task, attr)
                
                if isinstance(value, list):
                    requirements.extend([str(r) for r in value])
                elif isinstance(value, str):
                    requirements.append(value)
        
        # Default requirement based on task type/description
        if not requirements:
            task_desc = self._get_task_id(framework_task).lower()
            
            # Infer skills from task description
            if any(word in task_desc for word in ['code', 'program', 'dev']):
                requirements.append('coding')
            elif any(word in task_desc for word in ['write', 'content', 'text']):
                requirements.append('writing')
            elif any(word in task_desc for word in ['analyze', 'data', 'research']):
                requirements.append('analysis')
            else:
                requirements.append('general')
        
        return list(set(requirements))
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            'framework': self.framework_name,
            'backend': self.config.backend,
            'agents_mapped': len(self._agent_mapping),
            'tasks_mapped': len(self._task_mapping),
            'callbacks_registered': sum(len(callbacks) for callbacks in self._callbacks.values()),
            'real_time_enabled': self.config.enable_real_time,
            'fallback_enabled': self.config.fallback_enabled
        }