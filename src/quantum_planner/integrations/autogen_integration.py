"""AutoGen integration for quantum-optimized multi-agent conversations."""

from typing import List, Any, Dict, Optional, Callable
import logging

from .base_integration import BaseIntegration, IntegrationConfig, IntegrationError
from quantum_planner.models import Agent, Task, Solution

logger = logging.getLogger(__name__)


class AutoGenScheduler(BaseIntegration):
    """AutoGen integration with quantum task optimization."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        super().__init__("AutoGen", config)
        
        # AutoGen-specific configuration
        self.preserve_conversation_flow = True
        self.enable_dynamic_assignment = True
        self.conversation_timeout = 300.0  # 5 minutes
        
    def extract_agents(self, autogen_agents: List[Any]) -> List[Agent]:
        """Extract quantum planner agents from AutoGen agents."""
        
        quantum_agents = []
        
        for i, autogen_agent in enumerate(autogen_agents):
            try:
                # Extract agent information from AutoGen agent
                agent_id = self._get_autogen_agent_id(autogen_agent)
                skills = self._extract_autogen_skills(autogen_agent)
                capacity = self._get_autogen_capacity(autogen_agent)
                
                # Create quantum planner agent
                quantum_agent = Agent(
                    agent_id=agent_id,
                    skills=skills,
                    capacity=capacity,
                    availability=1.0,
                    preferences={
                        'autogen_index': i,
                        'agent_type': self._get_autogen_agent_type(autogen_agent)
                    }
                )
                
                quantum_agents.append(quantum_agent)
                self._agent_mapping[agent_id] = autogen_agent
                
            except Exception as e:
                self.logger.warning(f"Failed to extract AutoGen agent {i}: {e}")
                
                # Create fallback agent
                fallback_agent = Agent(
                    agent_id=f"autogen_agent_{i}",
                    skills=["conversation"],
                    capacity=1,
                    preferences={'autogen_index': i}
                )
                quantum_agents.append(fallback_agent)
                self._agent_mapping[f"autogen_agent_{i}"] = autogen_agent
        
        self.logger.info(f"Extracted {len(quantum_agents)} agents from AutoGen")
        return quantum_agents
    
    def extract_tasks(self, conversation_tasks: List[Any]) -> List[Task]:
        """Extract quantum planner tasks from AutoGen conversation tasks."""
        
        quantum_tasks = []
        
        for i, conv_task in enumerate(conversation_tasks):
            try:
                # Extract task information
                task_id = self._get_autogen_task_id(conv_task, i)
                required_skills = self._extract_autogen_task_requirements(conv_task)
                priority = self._get_autogen_priority(conv_task)
                duration = self._get_autogen_duration(conv_task)
                dependencies = self._get_autogen_dependencies(conv_task, conversation_tasks)
                
                # Create quantum planner task
                quantum_task = Task(
                    task_id=task_id,
                    required_skills=required_skills,
                    priority=priority,
                    duration=duration,
                    dependencies=dependencies
                )
                
                quantum_tasks.append(quantum_task)
                self._task_mapping[task_id] = conv_task
                
            except Exception as e:
                self.logger.warning(f"Failed to extract AutoGen task {i}: {e}")
                
                # Create fallback task
                fallback_task = Task(
                    task_id=f"autogen_task_{i}",
                    required_skills=["conversation"],
                    priority=1,
                    duration=1
                )
                quantum_tasks.append(fallback_task)
                self._task_mapping[f"autogen_task_{i}"] = conv_task
        
        self.logger.info(f"Extracted {len(quantum_tasks)} tasks from AutoGen")
        return quantum_tasks
    
    def apply_assignments(
        self,
        solution: Solution,
        autogen_agents: List[Any],
        conversation_tasks: List[Any]
    ) -> Dict[str, Any]:
        """Apply quantum optimization results to AutoGen conversation flow."""
        
        try:
            assignment_results = {}
            conversation_plan = []
            agent_roles = {}
            
            for task_id, agent_id in solution.assignments.items():
                # Get original AutoGen objects
                conv_task = self._task_mapping.get(task_id)
                autogen_agent = self._agent_mapping.get(agent_id)
                
                if conv_task and autogen_agent:
                    # Create conversation step
                    conversation_step = {
                        'task': conv_task,
                        'agent': autogen_agent,
                        'task_id': task_id,
                        'agent_id': agent_id,
                        'message_type': self._determine_message_type(conv_task),
                        'response_required': self._requires_response(conv_task)
                    }
                    
                    conversation_plan.append(conversation_step)
                    
                    # Track agent roles
                    if agent_id not in agent_roles:
                        agent_roles[agent_id] = {
                            'agent': autogen_agent,
                            'tasks': [],
                            'primary_role': self._get_autogen_agent_type(autogen_agent)
                        }
                    
                    agent_roles[agent_id]['tasks'].append(task_id)
                    
                    assignment_results[task_id] = {
                        'agent_id': agent_id,
                        'conversation_role': self._get_conversation_role(conv_task, autogen_agent)
                    }
            
            # Create optimized conversation flow
            if self.enable_dynamic_assignment:
                conversation_flow = self._create_dynamic_conversation_flow(
                    conversation_plan, solution, agent_roles
                )
            else:
                conversation_flow = self._create_static_conversation_flow(
                    conversation_plan, solution
                )
            
            return {
                'assignments': assignment_results,
                'conversation_plan': conversation_plan,
                'conversation_flow': conversation_flow,
                'agent_roles': agent_roles,
                'optimization_metrics': {
                    'makespan': solution.makespan,
                    'cost': solution.cost,
                    'backend_used': solution.backend_used,
                    'quality_score': solution.calculate_quality_score()
                }
            }
            
        except Exception as e:
            raise IntegrationError(f"Failed to apply assignments to AutoGen: {e}") from e
    
    def _get_autogen_agent_id(self, autogen_agent: Any) -> str:
        """Extract agent ID from AutoGen agent."""
        
        # Try AutoGen-specific attributes
        for attr in ['name', 'role', 'id']:
            if hasattr(autogen_agent, attr):
                value = getattr(autogen_agent, attr)
                if value:
                    return str(value)
        
        # Check for system message or description
        if hasattr(autogen_agent, 'system_message'):
            system_msg = getattr(autogen_agent, 'system_message', '')
            if system_msg:
                # Extract role from system message
                words = str(system_msg).split()[:3]
                return '_'.join(words).lower()
        
        # Fallback
        return f"autogen_agent_{id(autogen_agent)}"
    
    def _get_autogen_agent_type(self, autogen_agent: Any) -> str:
        """Determine AutoGen agent type."""
        
        agent_class = type(autogen_agent).__name__.lower()
        
        if 'assistant' in agent_class:
            return 'assistant'
        elif 'user' in agent_class:
            return 'user_proxy'
        elif 'groupchat' in agent_class:
            return 'group_chat_manager'
        elif 'retrieval' in agent_class:
            return 'retrieval_augmented'
        else:
            return 'conversational'
    
    def _extract_autogen_skills(self, autogen_agent: Any) -> List[str]:
        """Extract skills from AutoGen agent."""
        
        skills = []
        
        # Extract from system message
        if hasattr(autogen_agent, 'system_message'):
            system_msg = str(getattr(autogen_agent, 'system_message', '')).lower()
            
            # Skill keywords to look for
            skill_patterns = {
                'coding': ['code', 'program', 'python', 'javascript', 'sql'],
                'analysis': ['analyze', 'data', 'research', 'investigate'],  
                'writing': ['write', 'document', 'report', 'content'],
                'qa': ['test', 'quality', 'review', 'validate'],
                'management': ['manage', 'coordinate', 'organize', 'plan'],
                'conversation': ['chat', 'discuss', 'communicate', 'talk']
            }
            
            for skill, keywords in skill_patterns.items():
                if any(keyword in system_msg for keyword in keywords):
                    skills.append(skill)
        
        # Check for specific AutoGen capabilities
        if hasattr(autogen_agent, 'code_execution_config'):
            skills.append('code_execution')
        
        if hasattr(autogen_agent, 'function_map'):
            skills.append('function_calling')
        
        # Default skills based on agent type
        agent_type = self._get_autogen_agent_type(autogen_agent)
        default_skills = {
            'assistant': ['conversation', 'assistance'],
            'user_proxy': ['user_interaction', 'coordination'],
            'group_chat_manager': ['moderation', 'coordination'],
            'retrieval_augmented': ['information_retrieval', 'research']
        }
        
        skills.extend(default_skills.get(agent_type, ['conversation']))
        
        return list(set(skills)) if skills else ['conversation']
    
    def _get_autogen_capacity(self, autogen_agent: Any) -> int:
        """Determine AutoGen agent capacity."""
        
        # Check for explicit capacity setting
        if hasattr(autogen_agent, 'max_consecutive_auto_reply'):
            return min(int(getattr(autogen_agent, 'max_consecutive_auto_reply', 1)), 5)
        
        # Base capacity on agent type
        agent_type = self._get_autogen_agent_type(autogen_agent)
        
        capacity_map = {
            'assistant': 3,
            'user_proxy': 2,
            'group_chat_manager': 5,
            'retrieval_augmented': 2,
            'conversational': 2
        }
        
        return capacity_map.get(agent_type, 2)
    
    def _get_autogen_task_id(self, conv_task: Any, index: int) -> str:
        """Extract task ID from AutoGen conversation task."""
        
        # Try different ways to identify the task
        if isinstance(conv_task, dict):
            if 'message' in conv_task:
                return f"msg_{hash(str(conv_task['message']))}"
            elif 'content' in conv_task:
                return f"content_{hash(str(conv_task['content']))}"
        
        if hasattr(conv_task, 'content'):
            content = str(getattr(conv_task, 'content', ''))[:50]
            return f"task_{hash(content)}"
        
        if isinstance(conv_task, str):
            return f"str_task_{hash(conv_task)}"
        
        return f"autogen_task_{index}"
    
    def _extract_autogen_task_requirements(self, conv_task: Any) -> List[str]:
        """Extract required skills from AutoGen conversation task."""
        
        requirements = []
        
        # Extract content to analyze
        content = ""
        if isinstance(conv_task, dict):
            content = str(conv_task.get('message', conv_task.get('content', '')))
        elif hasattr(conv_task, 'content'):
            content = str(getattr(conv_task, 'content', ''))
        elif isinstance(conv_task, str):
            content = conv_task
        
        content_lower = content.lower()
        
        # Analyze content for skill requirements
        skill_indicators = {
            'coding': ['code', 'python', 'javascript', 'sql', 'algorithm', 'debug'],
            'analysis': ['analyze', 'data', 'statistics', 'research', 'investigate'],
            'writing': ['write', 'document', 'report', 'summary', 'explain'],
            'qa': ['test', 'review', 'validate', 'check', 'verify'],
            'creative': ['design', 'creative', 'brainstorm', 'ideate'],
            'conversation': ['discuss', 'chat', 'talk', 'communicate']
        }
        
        for skill, indicators in skill_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                requirements.append(skill)
        
        # Default requirement
        if not requirements:
            requirements = ['conversation']
        
        return requirements
    
    def _get_autogen_priority(self, conv_task: Any) -> int:
        """Determine AutoGen task priority."""
        
        # Extract content for priority analysis
        content = ""
        if isinstance(conv_task, dict):
            content = str(conv_task.get('message', conv_task.get('content', '')))
        elif hasattr(conv_task, 'content'):
            content = str(getattr(conv_task, 'content', ''))
        elif isinstance(conv_task, str):
            content = conv_task
        
        content_lower = content.lower()
        
        # Priority indicators
        if any(word in content_lower for word in ['urgent', 'critical', 'emergency']):
            return 5
        elif any(word in content_lower for word in ['important', 'priority', 'asap']):
            return 4
        elif any(word in content_lower for word in ['high', 'soon']):
            return 3
        elif any(word in content_lower for word in ['low', 'later', 'optional']):
            return 1
        else:
            return 2  # Default priority
    
    def _get_autogen_duration(self, conv_task: Any) -> int:
        """Estimate AutoGen task duration."""
        
        # Extract content length as complexity indicator
        content = ""
        if isinstance(conv_task, dict):
            content = str(conv_task.get('message', conv_task.get('content', '')))
        elif hasattr(conv_task, 'content'):
            content = str(getattr(conv_task, 'content', ''))
        elif isinstance(conv_task, str):
            content = conv_task
        
        # Base duration on content length and complexity
        base_duration = max(1, len(content) // 200)  # ~200 chars per duration unit
        
        content_lower = content.lower()
        
        # Complexity multipliers
        if any(word in content_lower for word in ['complex', 'detailed', 'comprehensive']):
            base_duration *= 2
        elif any(word in content_lower for word in ['simple', 'quick', 'brief']):
            base_duration = max(1, base_duration // 2)
        
        return min(base_duration, 8)  # Cap at 8 units
    
    def _get_autogen_dependencies(
        self, 
        conv_task: Any, 
        all_tasks: List[Any]
    ) -> List[str]:
        """Extract task dependencies from AutoGen conversation flow."""
        
        # AutoGen tasks typically have temporal dependencies
        # For now, return empty dependencies - could be enhanced with
        # conversation flow analysis
        
        return []
    
    def _determine_message_type(self, conv_task: Any) -> str:
        """Determine the type of message for conversation flow."""
        
        content = ""
        if isinstance(conv_task, dict):
            content = str(conv_task.get('message', conv_task.get('content', '')))
        elif hasattr(conv_task, 'content'):
            content = str(getattr(conv_task, 'content', ''))
        elif isinstance(conv_task, str):
            content = conv_task
        
        content_lower = content.lower()
        
        # Classify message type
        if '?' in content:
            return 'question'
        elif any(word in content_lower for word in ['please', 'can you', 'could you']):
            return 'request'
        elif content_lower.startswith(('here', 'this', 'i have')):
            return 'information'
        elif any(word in content_lower for word in ['thanks', 'thank you', 'good']):
            return 'acknowledgment'
        else:
            return 'statement'
    
    def _requires_response(self, conv_task: Any) -> bool:
        """Determine if conversation task requires a response."""
        
        message_type = self._determine_message_type(conv_task)
        
        response_required = {
            'question': True,
            'request': True,
            'information': False,
            'acknowledgment': False,
            'statement': True  # Most statements expect some response
        }
        
        return response_required.get(message_type, True)
    
    def _get_conversation_role(self, conv_task: Any, autogen_agent: Any) -> str:
        """Determine conversation role for this task-agent pair."""
        
        agent_type = self._get_autogen_agent_type(autogen_agent)
        message_type = self._determine_message_type(conv_task)
        
        # Define conversation roles
        role_matrix = {
            ('assistant', 'question'): 'responder',
            ('assistant', 'request'): 'executor',
            ('user_proxy', 'question'): 'questioner',
            ('user_proxy', 'request'): 'requester',
            ('group_chat_manager', 'question'): 'moderator',
            ('group_chat_manager', 'request'): 'coordinator'
        }
        
        return role_matrix.get((agent_type, message_type), 'participant')
    
    def _create_dynamic_conversation_flow(
        self,
        conversation_plan: List[Dict[str, Any]],
        solution: Solution,
        agent_roles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create dynamic conversation flow with adaptive scheduling."""
        
        return {
            'type': 'dynamic',
            'flow_control': 'adaptive',
            'max_turns': 50,
            'timeout': self.conversation_timeout,
            'speaker_selection': 'quantum_optimized',
            'parallel_conversations': len(agent_roles) > 2,
            'conversation_order': [step['task_id'] for step in conversation_plan],
            'checkpoints': [step['task_id'] for step in conversation_plan[::5]]  # Every 5th task
        }
    
    def _create_static_conversation_flow(
        self,
        conversation_plan: List[Dict[str, Any]],
        solution: Solution
    ) -> Dict[str, Any]:
        """Create static conversation flow with predetermined order."""
        
        return {
            'type': 'static',
            'flow_control': 'sequential',
            'conversation_order': [step['task_id'] for step in conversation_plan],
            'speaker_transitions': self._create_speaker_transitions(conversation_plan)
        }
    
    def _create_speaker_transitions(
        self,
        conversation_plan: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Create speaker transition plan."""
        
        transitions = []
        
        for i in range(len(conversation_plan) - 1):
            current_step = conversation_plan[i]
            next_step = conversation_plan[i + 1]
            
            transitions.append({
                'from_agent': current_step['agent_id'],
                'to_agent': next_step['agent_id'],
                'from_task': current_step['task_id'],
                'to_task': next_step['task_id']
            })
        
        return transitions
    
    def create_optimized_group_chat(
        self,
        agents: List[Any],
        messages: List[Any],
        admin_agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Create an optimized AutoGen GroupChat with quantum scheduling."""
        
        try:
            # Convert messages to tasks
            tasks = messages
            
            # Run optimization
            result = self.optimize_framework_tasks(agents, tasks)
            
            if not result['success']:
                self.logger.warning("Optimization failed for group chat")
                return {'success': False, 'error': 'Optimization failed'}
            
            # Create group chat configuration
            group_chat_config = {
                'agents': agents,
                'messages': messages,
                'admin_name': getattr(admin_agent, 'name', 'Admin') if admin_agent else 'Admin',
                'max_round': min(50, len(messages) * 2),
                'speaker_selection_method': 'quantum_optimized',
                'optimization_result': result
            }
            
            return {
                'success': True,
                'group_chat_config': group_chat_config,
                'optimization_metrics': result.get('optimization_metrics', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create optimized group chat: {e}")
            return {'success': False, 'error': str(e)}