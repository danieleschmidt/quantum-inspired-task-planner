"""CrewAI integration for quantum-optimized task scheduling."""

from typing import List, Any, Dict, Optional
import logging

from .base_integration import BaseIntegration, IntegrationConfig, IntegrationError
from quantum_planner.models import Agent, Task, Solution

logger = logging.getLogger(__name__)


class CrewAIScheduler(BaseIntegration):
    """CrewAI integration with quantum task optimization."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        super().__init__("CrewAI", config)
        
        # CrewAI-specific configuration
        self.preserve_crew_structure = True
        self.respect_agent_roles = True
        
    def extract_agents(self, crew_agents: List[Any]) -> List[Agent]:
        """Extract quantum planner agents from CrewAI agents."""
        
        quantum_agents = []
        
        for i, crew_agent in enumerate(crew_agents):
            try:
                # Extract agent information from CrewAI agent
                agent_id = self._get_crewai_agent_id(crew_agent)
                skills = self._extract_crewai_skills(crew_agent)
                capacity = self._get_crewai_capacity(crew_agent)
                
                # Create quantum planner agent
                quantum_agent = Agent(
                    agent_id=agent_id,
                    skills=skills,
                    capacity=capacity,
                    availability=1.0,  # Assume full availability
                    preferences={'crewai_index': i}
                )
                
                quantum_agents.append(quantum_agent)
                self._agent_mapping[agent_id] = crew_agent
                
            except Exception as e:
                self.logger.warning(f"Failed to extract CrewAI agent {i}: {e}")
                
                # Create fallback agent
                fallback_agent = Agent(
                    agent_id=f"crewai_agent_{i}",
                    skills=["general"],
                    capacity=1,
                    preferences={'crewai_index': i}
                )
                quantum_agents.append(fallback_agent)
                self._agent_mapping[f"crewai_agent_{i}"] = crew_agent
        
        self.logger.info(f"Extracted {len(quantum_agents)} agents from CrewAI")
        return quantum_agents
    
    def extract_tasks(self, crew_tasks: List[Any]) -> List[Task]:
        """Extract quantum planner tasks from CrewAI tasks."""
        
        quantum_tasks = []
        
        for i, crew_task in enumerate(crew_tasks):
            try:
                # Extract task information from CrewAI task
                task_id = self._get_crewai_task_id(crew_task, i)
                required_skills = self._extract_crewai_task_requirements(crew_task)
                priority = self._get_crewai_priority(crew_task)
                duration = self._get_crewai_duration(crew_task)
                dependencies = self._get_crewai_dependencies(crew_task, crew_tasks)
                
                # Create quantum planner task
                quantum_task = Task(
                    task_id=task_id,
                    required_skills=required_skills,
                    priority=priority,
                    duration=duration,
                    dependencies=dependencies
                )
                
                quantum_tasks.append(quantum_task)
                self._task_mapping[task_id] = crew_task
                
            except Exception as e:
                self.logger.warning(f"Failed to extract CrewAI task {i}: {e}")
                
                # Create fallback task
                fallback_task = Task(
                    task_id=f"crewai_task_{i}",
                    required_skills=["general"],
                    priority=1,
                    duration=1
                )
                quantum_tasks.append(fallback_task)
                self._task_mapping[f"crewai_task_{i}"] = crew_task
        
        self.logger.info(f"Extracted {len(quantum_tasks)} tasks from CrewAI")
        return quantum_tasks
    
    def apply_assignments(
        self,
        solution: Solution,
        crew_agents: List[Any],
        crew_tasks: List[Any]
    ) -> Dict[str, Any]:
        """Apply quantum optimization results to CrewAI crew."""
        
        try:
            assignment_results = {}
            execution_plan = []
            
            for task_id, agent_id in solution.assignments.items():
                # Get original CrewAI objects
                crew_task = self._task_mapping.get(task_id)
                crew_agent = self._agent_mapping.get(agent_id)
                
                if crew_task and crew_agent:
                    # Apply assignment to CrewAI task
                    if hasattr(crew_task, 'agent') and crew_task.agent != crew_agent:
                        # Reassign agent if different
                        original_agent = getattr(crew_task, 'agent', None)
                        crew_task.agent = crew_agent
                        
                        assignment_results[task_id] = {
                            'agent_id': agent_id,
                            'original_agent': self._get_crewai_agent_id(original_agent) if original_agent else None,
                            'new_agent': agent_id,
                            'reassigned': True
                        }
                    else:
                        assignment_results[task_id] = {
                            'agent_id': agent_id,
                            'reassigned': False
                        }
                    
                    # Add to execution plan
                    execution_plan.append({
                        'task': crew_task,
                        'agent': crew_agent,
                        'task_id': task_id,
                        'agent_id': agent_id
                    })
                else:
                    self.logger.warning(f"Could not find CrewAI objects for assignment: {task_id} -> {agent_id}")
            
            # Create optimized crew execution strategy
            if self.config.enable_real_time:
                execution_strategy = self._create_real_time_strategy(execution_plan, solution)
            else:
                execution_strategy = self._create_batch_strategy(execution_plan, solution)
            
            return {
                'assignments': assignment_results,
                'execution_plan': execution_plan,
                'execution_strategy': execution_strategy,
                'optimization_metrics': {
                    'makespan': solution.makespan,
                    'cost': solution.cost,
                    'backend_used': solution.backend_used,
                    'quality_score': solution.calculate_quality_score()
                }
            }
            
        except Exception as e:
            raise IntegrationError(f"Failed to apply assignments to CrewAI: {e}") from e
    
    def _get_crewai_agent_id(self, crew_agent: Any) -> str:
        """Extract agent ID from CrewAI agent."""
        
        # Try CrewAI-specific attributes
        for attr in ['role', 'id', 'name']:
            if hasattr(crew_agent, attr):
                value = getattr(crew_agent, attr)
                if value:
                    return str(value)
        
        # Fallback
        return f"crewai_agent_{id(crew_agent)}"
    
    def _extract_crewai_skills(self, crew_agent: Any) -> List[str]:
        """Extract skills from CrewAI agent."""
        
        skills = []
        
        # CrewAI-specific skill extraction
        if hasattr(crew_agent, 'role'):
            skills.append(str(crew_agent.role))
        
        if hasattr(crew_agent, 'tools'):
            tools = getattr(crew_agent, 'tools', [])
            if tools:
                skills.extend([str(tool) for tool in tools])
        
        if hasattr(crew_agent, 'backstory'):
            backstory = getattr(crew_agent, 'backstory', '')
            # Extract skills from backstory (simple keyword matching)
            skill_keywords = ['python', 'javascript', 'writing', 'analysis', 'research', 'design']
            for keyword in skill_keywords:
                if keyword.lower() in backstory.lower():
                    skills.append(keyword)
        
        # Ensure at least one skill
        if not skills:
            skills = ['general']
        
        return list(set(skills))
    
    def _get_crewai_capacity(self, crew_agent: Any) -> int:
        """Determine CrewAI agent capacity."""
        
        # Try to get capacity from agent attributes
        if hasattr(crew_agent, 'max_tasks'):
            return int(getattr(crew_agent, 'max_tasks'))
        
        if hasattr(crew_agent, 'capacity'):
            return int(getattr(crew_agent, 'capacity'))
        
        # Default capacity based on agent role/tools
        tools_count = len(getattr(crew_agent, 'tools', []))
        base_capacity = max(1, tools_count // 2 + 1)
        
        return min(base_capacity, 5)  # Cap at 5
    
    def _get_crewai_task_id(self, crew_task: Any, index: int) -> str:
        """Extract task ID from CrewAI task."""
        
        for attr in ['id', 'description']:
            if hasattr(crew_task, attr):
                value = getattr(crew_task, attr)
                if value:
                    return str(value)[:50]  # Limit length
        
        return f"crewai_task_{index}"
    
    def _extract_crewai_task_requirements(self, crew_task: Any) -> List[str]:
        """Extract required skills from CrewAI task."""
        
        requirements = []
        
        # Extract from task description
        if hasattr(crew_task, 'description'):
            description = str(getattr(crew_task, 'description', '')).lower()
            
            # Simple keyword-based skill inference
            skill_map = {
                'code': ['coding', 'programming'],
                'write': ['writing', 'content'],
                'analyze': ['analysis', 'research'],
                'design': ['design', 'creative'],
                'test': ['testing', 'qa'],
                'deploy': ['deployment', 'devops']
            }
            
            for keyword, skills in skill_map.items():
                if keyword in description:
                    requirements.extend(skills)
        
        # Extract from expected output type
        if hasattr(crew_task, 'expected_output'):
            output_type = str(getattr(crew_task, 'expected_output', '')).lower()
            if 'code' in output_type:
                requirements.append('coding')
            elif 'report' in output_type or 'document' in output_type:
                requirements.append('writing')
        
        # Ensure at least one requirement
        if not requirements:
            requirements = ['general']
        
        return list(set(requirements))
    
    def _get_crewai_priority(self, crew_task: Any) -> int:
        """Determine CrewAI task priority."""
        
        if hasattr(crew_task, 'priority'):
            return int(getattr(crew_task, 'priority', 1))
        
        # Infer priority from task attributes
        priority = 1
        
        if hasattr(crew_task, 'description'):
            description = str(getattr(crew_task, 'description', '')).lower()
            if any(word in description for word in ['urgent', 'critical', 'important']):
                priority = 5
            elif any(word in description for word in ['high', 'priority']):
                priority = 3
        
        return priority
    
    def _get_crewai_duration(self, crew_task: Any) -> int:
        """Estimate CrewAI task duration."""
        
        if hasattr(crew_task, 'duration'):
            return int(getattr(crew_task, 'duration', 1))
        
        # Estimate based on task complexity
        duration = 1
        
        if hasattr(crew_task, 'description'):
            description = str(getattr(crew_task, 'description', ''))
            
            # Simple heuristic based on description length and complexity keywords
            base_duration = max(1, len(description) // 100)
            
            complexity_keywords = ['complex', 'detailed', 'comprehensive', 'thorough']
            if any(keyword in description.lower() for keyword in complexity_keywords):
                base_duration *= 2
            
            duration = min(base_duration, 10)  # Cap at 10
        
        return duration
    
    def _get_crewai_dependencies(self, crew_task: Any, all_tasks: List[Any]) -> List[str]:
        """Extract task dependencies from CrewAI task."""
        
        dependencies = []
        
        if hasattr(crew_task, 'dependencies'):
            deps = getattr(crew_task, 'dependencies', [])
            for dep in deps:
                if hasattr(dep, 'id'):
                    dependencies.append(str(dep.id))
                else:
                    dependencies.append(str(dep))
        
        # Infer dependencies from task context (optional)
        if hasattr(crew_task, 'context') and hasattr(crew_task, 'description'):
            # Simple dependency inference could be added here
            pass
        
        return dependencies
    
    def _create_real_time_strategy(
        self,
        execution_plan: List[Dict[str, Any]],
        solution: Solution
    ) -> Dict[str, Any]:
        """Create real-time execution strategy for CrewAI."""
        
        return {
            'type': 'real_time',
            'update_interval': self.config.update_interval,
            'monitoring_enabled': True,
            'adaptive_scheduling': True,
            'parallel_execution': True,
            'execution_order': [item['task_id'] for item in execution_plan]
        }
    
    def _create_batch_strategy(
        self,
        execution_plan: List[Dict[str, Any]],
        solution: Solution
    ) -> Dict[str, Any]:
        """Create batch execution strategy for CrewAI."""
        
        return {
            'type': 'batch',
            'execution_order': [item['task_id'] for item in execution_plan],
            'parallel_execution': False,
            'checkpoint_enabled': True
        }
    
    def create_optimized_crew(
        self,
        original_crew: Any,
        agents: List[Any],
        tasks: List[Any]
    ) -> Any:
        """Create an optimized CrewAI crew with quantum scheduling."""
        
        try:
            # Run optimization
            result = self.optimize_framework_tasks(agents, tasks)
            
            if not result['success']:
                self.logger.warning("Optimization failed, returning original crew")
                return original_crew
            
            # The assignments have already been applied to the tasks
            # Return the original crew with optimized task assignments
            self.logger.info("Created optimized CrewAI crew")
            return original_crew
            
        except Exception as e:
            self.logger.error(f"Failed to create optimized crew: {e}")
            return original_crew