#!/usr/bin/env python3
"""
Robust Quantum Task Planner - Generation 2: Make It Robust
Enhanced with comprehensive error handling, validation, monitoring, and resilience.
"""

from typing import List, Dict, Optional, Any, Union, Callable
import sys
import os
import time
import logging
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_planner import QuantumTaskPlanner, Agent, Task, Solution


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


@dataclass
class PlannerError:
    """Structured error information."""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_action: Optional[str] = None


@dataclass 
class HealthMetrics:
    """Health monitoring metrics."""
    success_rate: float = 0.0
    average_solve_time: float = 0.0
    total_assignments: int = 0
    error_count: int = 0
    last_success: Optional[float] = None
    backend_availability: Dict[str, bool] = field(default_factory=dict)


class RobustQuantumPlanner:
    """
    Production-ready quantum task planner with comprehensive error handling,
    validation, monitoring, logging, and resilience features.
    """
    
    def __init__(
        self, 
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        enable_monitoring: bool = True,
        enable_caching: bool = True,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        log_level: str = "INFO"
    ):
        """
        Initialize robust planner with enhanced capabilities.
        
        Args:
            validation_level: How strict to be with input validation
            enable_monitoring: Whether to collect performance metrics
            enable_caching: Whether to cache solutions for repeated problems
            timeout_seconds: Maximum time to spend on optimization
            max_retries: Number of retry attempts on failure
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.validation_level = validation_level
        self.enable_monitoring = enable_monitoring
        self.enable_caching = enable_caching
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Setup structured logging
        self._setup_logging(log_level)
        
        # Initialize core planner with multiple fallback options
        self._initialize_planner_chain()
        
        # Initialize monitoring and metrics
        self.health_metrics = HealthMetrics()
        self.error_history: List[PlannerError] = []
        self.solution_cache: Dict[str, Dict] = {}
        
        # Performance tracking
        self.assignment_times: List[float] = []
        self.success_count = 0
        self.failure_count = 0
        
        self.logger.info("RobustQuantumPlanner initialized successfully", extra={
            'validation_level': validation_level.value,
            'monitoring_enabled': enable_monitoring,
            'caching_enabled': enable_caching,
            'timeout': timeout_seconds
        })
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup structured logging with context."""
        self.logger = logging.getLogger(f"RobustQuantumPlanner.{id(self)}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter with context
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _initialize_planner_chain(self) -> None:
        """Initialize planner with fallback chain for maximum reliability."""
        try:
            # Try quantum-enabled planner first
            self.primary_planner = QuantumTaskPlanner(
                backend="auto", 
                fallback="simulated_annealing"
            )
            self.logger.info("Primary quantum planner initialized")
        except Exception as e:
            self.logger.warning(f"Quantum planner failed to initialize: {e}")
            
        try:
            # Always have classical fallback available  
            self.fallback_planner = QuantumTaskPlanner(
                backend="simulated_annealing",
                fallback=None
            )
            self.logger.info("Classical fallback planner initialized")
        except Exception as e:
            self.logger.error(f"Even classical planner failed: {e}")
            raise RuntimeError("Cannot initialize any planner backend") from e
    
    @contextmanager
    def _error_handling_context(self, operation: str):
        """Context manager for consistent error handling."""
        start_time = time.time()
        try:
            self.logger.debug(f"Starting operation: {operation}")
            yield
            
            # Record success
            duration = time.time() - start_time
            self.assignment_times.append(duration)
            self.success_count += 1
            
            if self.enable_monitoring:
                self.health_metrics.last_success = time.time()
                self.health_metrics.total_assignments += 1
                self._update_success_rate()
            
            self.logger.info(f"Operation {operation} completed successfully in {duration:.2f}s")
            
        except Exception as e:
            # Record failure with detailed context
            duration = time.time() - start_time
            self.failure_count += 1
            
            # Create structured error
            error = PlannerError(
                error_type=type(e).__name__,
                message=str(e),
                severity=self._classify_error_severity(e),
                timestamp=time.time(),
                context={
                    'operation': operation,
                    'duration': duration,
                    'attempt_number': getattr(e, 'attempt_number', 1)
                },
                suggested_action=self._get_suggested_action(e)
            )
            
            self.error_history.append(error)
            
            # Log with appropriate level based on severity
            log_level = {
                ErrorSeverity.LOW: self.logger.info,
                ErrorSeverity.MEDIUM: self.logger.warning,
                ErrorSeverity.HIGH: self.logger.error,
                ErrorSeverity.CRITICAL: self.logger.critical
            }[error.severity]
            
            log_level(
                f"Operation {operation} failed: {error.message}",
                extra={
                    'error_type': error.error_type,
                    'severity': error.severity.value,
                    'suggested_action': error.suggested_action,
                    'context': error.context
                }
            )
            
            # Update health metrics
            if self.enable_monitoring:
                self.health_metrics.error_count += 1
                self._update_success_rate()
            
            raise
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity for appropriate handling."""
        if isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.HIGH  # Input validation errors are serious
        elif isinstance(error, TimeoutError):
            return ErrorSeverity.MEDIUM  # Timeouts are concerning but recoverable
        elif isinstance(error, ConnectionError):
            return ErrorSeverity.MEDIUM  # Network issues are often transient
        elif isinstance(error, RuntimeError):
            return ErrorSeverity.HIGH  # Runtime errors need investigation
        else:
            return ErrorSeverity.MEDIUM  # Default to medium for unknown errors
    
    def _get_suggested_action(self, error: Exception) -> str:
        """Provide actionable suggestions based on error type."""
        suggestions = {
            'ValueError': 'Check input data format and constraints',
            'TimeoutError': 'Try reducing problem size or increasing timeout',
            'ConnectionError': 'Check network connectivity and backend availability',  
            'ModuleNotFoundError': 'Install required dependencies',
            'RuntimeError': 'Review logs for detailed error context'
        }
        return suggestions.get(type(error).__name__, 'Review error details and logs')
    
    def _update_success_rate(self) -> None:
        """Update success rate metric."""
        total = self.success_count + self.failure_count
        if total > 0:
            self.health_metrics.success_rate = self.success_count / total
        
        if self.assignment_times:
            self.health_metrics.average_solve_time = sum(self.assignment_times) / len(self.assignment_times)
    
    def _validate_agents(self, agents: List[Dict[str, Any]]) -> List[str]:
        """Comprehensive agent validation with detailed error reporting."""
        errors = []
        
        if not agents:
            errors.append("No agents provided")
            return errors
            
        for i, agent in enumerate(agents):
            agent_context = f"Agent {i}"
            
            # Required fields
            if 'id' not in agent:
                errors.append(f"{agent_context}: Missing required 'id' field")
                continue
                
            if 'skills' not in agent:
                errors.append(f"{agent_context} ({agent['id']}): Missing required 'skills' field")
                continue
                
            # Validate data types
            if not isinstance(agent['id'], str) or not agent['id'].strip():
                errors.append(f"{agent_context}: 'id' must be a non-empty string")
                
            if not isinstance(agent['skills'], list):
                errors.append(f"{agent_context} ({agent['id']}): 'skills' must be a list")
            elif not agent['skills']:
                errors.append(f"{agent_context} ({agent['id']}): 'skills' cannot be empty")
            elif not all(isinstance(skill, str) and skill.strip() for skill in agent['skills']):
                errors.append(f"{agent_context} ({agent['id']}): All skills must be non-empty strings")
                
            # Validate capacity
            capacity = agent.get('capacity', 1)
            if not isinstance(capacity, (int, float)) or capacity <= 0:
                errors.append(f"{agent_context} ({agent['id']}): 'capacity' must be positive number")
                
            # Advanced validations for strict/comprehensive levels
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                # Check for duplicate agent IDs
                agent_ids = [a.get('id') for a in agents if 'id' in a]
                if agent_ids.count(agent['id']) > 1:
                    errors.append(f"{agent_context} ({agent['id']}): Duplicate agent ID detected")
                    
                # Check for very long skill names (might indicate data corruption)
                if 'skills' in agent and isinstance(agent['skills'], list):
                    for skill in agent['skills']:
                        if isinstance(skill, str) and len(skill) > 100:
                            errors.append(f"{agent_context} ({agent['id']}): Skill name too long: '{skill[:50]}...'")
            
            # Comprehensive validations
            if self.validation_level == ValidationLevel.COMPREHENSIVE:
                # Check for reasonable capacity limits
                if capacity > 1000:
                    errors.append(f"{agent_context} ({agent['id']}): Capacity {capacity} seems unreasonably high")
                    
                # Check for availability if provided
                availability = agent.get('availability', 1.0)
                if not isinstance(availability, (int, float)) or not 0 <= availability <= 1:
                    errors.append(f"{agent_context} ({agent['id']}): 'availability' must be between 0 and 1")
        
        return errors
    
    def _validate_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Comprehensive task validation with detailed error reporting."""
        errors = []
        
        if not tasks:
            errors.append("No tasks provided")
            return errors
            
        for i, task in enumerate(tasks):
            task_context = f"Task {i}"
            
            # Required fields
            if 'id' not in task:
                errors.append(f"{task_context}: Missing required 'id' field")
                continue
                
            if 'skills' not in task:
                errors.append(f"{task_context} ({task['id']}): Missing required 'skills' field")
                continue
                
            # Validate data types
            if not isinstance(task['id'], str) or not task['id'].strip():
                errors.append(f"{task_context}: 'id' must be a non-empty string")
                
            if not isinstance(task['skills'], list):
                errors.append(f"{task_context} ({task['id']}): 'skills' must be a list")
            elif not task['skills']:
                errors.append(f"{task_context} ({task['id']}): 'skills' cannot be empty")
            elif not all(isinstance(skill, str) and skill.strip() for skill in task['skills']):
                errors.append(f"{task_context} ({task['id']}): All skills must be non-empty strings")
                
            # Validate optional fields
            priority = task.get('priority', 1)
            if not isinstance(priority, (int, float)) or priority <= 0:
                errors.append(f"{task_context} ({task['id']}): 'priority' must be positive number")
                
            duration = task.get('duration', 1)  
            if not isinstance(duration, (int, float)) or duration <= 0:
                errors.append(f"{task_context} ({task['id']}): 'duration' must be positive number")
                
            # Advanced validations
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
                # Check for duplicate task IDs
                task_ids = [t.get('id') for t in tasks if 'id' in t]
                if task_ids.count(task['id']) > 1:
                    errors.append(f"{task_context} ({task['id']}): Duplicate task ID detected")
                    
                # Check dependencies if provided
                dependencies = task.get('dependencies', [])
                if dependencies and not isinstance(dependencies, list):
                    errors.append(f"{task_context} ({task['id']}): 'dependencies' must be a list")
                elif isinstance(dependencies, list):
                    for dep in dependencies:
                        if not isinstance(dep, str) or not dep.strip():
                            errors.append(f"{task_context} ({task['id']}): All dependencies must be non-empty strings")
                        elif dep == task['id']:
                            errors.append(f"{task_context} ({task['id']}): Task cannot depend on itself")
            
            # Comprehensive validations
            if self.validation_level == ValidationLevel.COMPREHENSIVE:
                # Check for reasonable duration limits
                if duration > 1000:
                    errors.append(f"{task_context} ({task['id']}): Duration {duration} seems unreasonably high")
                    
                # Check for reasonable priority limits  
                if priority > 1000:
                    errors.append(f"{task_context} ({task['id']}): Priority {priority} seems unreasonably high")
        
        return errors
    
    def _validate_assignment_feasibility(
        self, 
        agents: List[Dict[str, Any]], 
        tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Check if task assignment is theoretically possible."""
        errors = []
        
        # Collect all agent skills
        all_agent_skills = set()
        for agent in agents:
            if 'skills' in agent and isinstance(agent['skills'], list):
                all_agent_skills.update(agent['skills'])
        
        # Check if all task skills are covered
        uncovered_skills = set()
        unassignable_tasks = []
        
        for task in tasks:
            if 'skills' in task and isinstance(task['skills'], list):
                task_skills = set(task['skills'])
                missing_skills = task_skills - all_agent_skills
                
                if missing_skills:
                    uncovered_skills.update(missing_skills)
                    unassignable_tasks.append(task['id'])
        
        if uncovered_skills:
            errors.append(
                f"Skills not available in any agent: {sorted(uncovered_skills)}. "
                f"This affects tasks: {unassignable_tasks}"
            )
        
        # Check total capacity vs task load (basic estimate)
        total_agent_capacity = sum(agent.get('capacity', 1) for agent in agents)
        total_task_load = sum(task.get('duration', 1) for task in tasks)
        
        if total_task_load > total_agent_capacity * 10:  # Allow some reasonable multiplier
            errors.append(
                f"Total task load ({total_task_load}) significantly exceeds "
                f"total agent capacity ({total_agent_capacity}). "
                f"Assignment may result in very high makespan."
            )
        
        return errors
    
    def _generate_cache_key(
        self, 
        agents: List[Dict[str, Any]], 
        tasks: List[Dict[str, Any]],
        minimize: str
    ) -> str:
        """Generate a cache key for solution caching."""
        # Create a simplified representation for caching
        agent_repr = tuple(sorted((
            agent['id'], 
            tuple(sorted(agent['skills'])), 
            agent.get('capacity', 1)
        ) for agent in agents))
        
        task_repr = tuple(sorted((
            task['id'],
            tuple(sorted(task['skills'])),
            task.get('priority', 1),
            task.get('duration', 1)
        ) for task in tasks))
        
        cache_data = {
            'agents': agent_repr,
            'tasks': task_repr,  
            'minimize': minimize
        }
        
        return str(hash(str(cache_data)))
    
    def assign_tasks_robust(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]], 
        minimize: str = "time",
        timeout_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Robust task assignment with comprehensive error handling and validation.
        
        Args:
            agents: List of agent dictionaries
            tasks: List of task dictionaries
            minimize: Optimization objective ('time' or 'cost')
            timeout_override: Override default timeout
            
        Returns:
            Detailed result dictionary with success status, assignments, metrics, and diagnostics
        """
        with self._error_handling_context("robust_task_assignment"):
            start_time = time.time()
            timeout = timeout_override or self.timeout_seconds
            
            # Step 1: Comprehensive Input Validation
            self.logger.debug("Starting input validation", extra={
                'validation_level': self.validation_level.value,
                'agent_count': len(agents),
                'task_count': len(tasks)
            })
            
            validation_errors = []
            validation_errors.extend(self._validate_agents(agents))
            validation_errors.extend(self._validate_tasks(tasks))
            validation_errors.extend(self._validate_assignment_feasibility(agents, tasks))
            
            if validation_errors:
                error_msg = "Input validation failed:\\n" + "\\n".join(f"- {error}" for error in validation_errors)
                raise ValueError(error_msg)
            
            self.logger.info("Input validation passed", extra={
                'agents': len(agents),
                'tasks': len(tasks),
                'validation_time': time.time() - start_time
            })
            
            # Step 2: Check cache if enabled
            cache_key = None
            if self.enable_caching:
                cache_key = self._generate_cache_key(agents, tasks, minimize)
                if cache_key in self.solution_cache:
                    cached_solution = self.solution_cache[cache_key]
                    cached_solution['cache_hit'] = True
                    cached_solution['retrieval_time'] = time.time() - start_time
                    
                    self.logger.info("Using cached solution", extra={
                        'cache_key': cache_key[:16] + "...",
                        'retrieval_time': cached_solution['retrieval_time']
                    })
                    
                    return cached_solution
            
            # Step 3: Execute assignment with retry logic
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    self.logger.debug(f"Assignment attempt {attempt + 1}/{self.max_retries}")
                    
                    # Try primary planner first
                    planner_to_use = self.primary_planner
                    planner_name = "primary"
                    
                    # If previous attempts failed, try fallback
                    if attempt > 0 and hasattr(self, 'fallback_planner'):
                        planner_to_use = self.fallback_planner  
                        planner_name = "fallback"
                    
                    # Convert to internal models
                    agent_objects = [Agent(
                        agent_id=a['id'],
                        skills=a['skills'],
                        capacity=a.get('capacity', 1),
                        availability=a.get('availability', 1.0),
                        cost_per_hour=a.get('cost_per_hour', 0.0)
                    ) for a in agents]
                    
                    task_objects = [Task(
                        task_id=t['id'],
                        required_skills=t['skills'],
                        priority=t.get('priority', 1),
                        duration=t.get('duration', 1),
                        dependencies=t.get('dependencies', [])
                    ) for t in tasks]
                    
                    # Execute with timeout
                    objective_map = {
                        'time': 'minimize_makespan',
                        'cost': 'minimize_cost'
                    }
                    objective = objective_map.get(minimize, 'minimize_makespan')
                    
                    assignment_start = time.time()
                    
                    # Add timeout handling  
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # Suppress quantum backend warnings during assignment
                        
                        solution = planner_to_use.assign(
                            agents=agent_objects,
                            tasks=task_objects,
                            objective=objective,
                            constraints={
                                'skill_match': True,
                                'capacity_limit': True
                            }
                        )
                    
                    assignment_time = time.time() - assignment_start
                    
                    # Check for timeout
                    if assignment_time > timeout:
                        raise TimeoutError(f"Assignment exceeded timeout of {timeout}s")
                    
                    # Step 4: Build comprehensive result
                    result = {
                        'success': True,
                        'assignments': solution.assignments,
                        'completion_time': solution.makespan,
                        'total_cost': solution.cost,
                        'backend_used': solution.backend_used,
                        'message': 'Tasks assigned successfully with robust validation',
                        
                        # Detailed metrics
                        'metrics': {
                            'assignment_time': assignment_time,
                            'validation_time': time.time() - start_time,
                            'total_time': time.time() - start_time,
                            'planner_used': planner_name,
                            'attempt_number': attempt + 1,
                            'timeout_used': timeout
                        },
                        
                        # Quality analysis
                        'quality_analysis': {
                            'quality_score': solution.calculate_quality_score(),
                            'load_distribution': solution.get_load_distribution(),
                            'assigned_agents': list(solution.get_assigned_agents()),
                            'task_count': solution.get_task_count()
                        },
                        
                        # Diagnostic information
                        'diagnostics': {
                            'validation_level': self.validation_level.value,
                            'cache_enabled': self.enable_caching,
                            'monitoring_enabled': self.enable_monitoring,
                            'agent_count': len(agents),
                            'task_count': len(tasks),
                            'solution_metadata': solution.metadata or {}
                        }
                    }
                    
                    # Step 5: Cache solution if enabled
                    if self.enable_caching and cache_key:
                        # Store a copy without cache_hit flag for future use
                        cache_result = result.copy()
                        cache_result.pop('cache_hit', None)
                        self.solution_cache[cache_key] = cache_result
                        result['cache_stored'] = True
                        
                        # Manage cache size (keep last 100 solutions)
                        if len(self.solution_cache) > 100:
                            oldest_key = next(iter(self.solution_cache))
                            del self.solution_cache[oldest_key]
                    
                    self.logger.info("Robust assignment completed successfully", extra={
                        'total_time': result['metrics']['total_time'],
                        'quality_score': result['quality_analysis']['quality_score'],
                        'planner_used': planner_name,
                        'attempt': attempt + 1
                    })
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    setattr(e, 'attempt_number', attempt + 1)  # Add attempt number to exception
                    
                    self.logger.warning(
                        f"Assignment attempt {attempt + 1} failed: {e}",
                        extra={
                            'attempt': attempt + 1,
                            'max_retries': self.max_retries,
                            'error_type': type(e).__name__,
                            'planner_used': planner_name
                        }
                    )
                    
                    # Don't retry for validation errors (they won't succeed)
                    if isinstance(e, ValueError):
                        break
                        
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s...
                        self.logger.debug(f"Waiting {wait_time}s before retry")
                        time.sleep(wait_time)
            
            # All attempts failed
            total_time = time.time() - start_time
            error_result = {
                'success': False,
                'assignments': {},
                'completion_time': 0,
                'total_cost': 0,
                'backend_used': 'none',
                'error': str(last_exception),
                'error_type': type(last_exception).__name__,
                'message': f'All {self.max_retries} assignment attempts failed',
                
                'metrics': {
                    'total_time': total_time,
                    'attempts_made': self.max_retries,
                    'timeout_used': timeout
                },
                
                'diagnostics': {
                    'last_error': str(last_exception),
                    'error_severity': self._classify_error_severity(last_exception).value,
                    'suggested_action': self._get_suggested_action(last_exception),
                    'validation_level': self.validation_level.value
                }
            }
            
            self.logger.error("All assignment attempts failed", extra=error_result)
            return error_result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health and performance status."""
        current_time = time.time()
        
        # Calculate recent error rate (last hour)
        recent_errors = [
            error for error in self.error_history 
            if current_time - error.timestamp < 3600
        ]
        
        # Backend availability check
        backend_status = {}
        try:
            if hasattr(self, 'primary_planner'):
                backend_status['primary'] = {
                    'available': True,
                    'type': 'quantum_enabled'
                }
        except:
            backend_status['primary'] = {
                'available': False,
                'type': 'unknown'
            }
            
        try:
            if hasattr(self, 'fallback_planner'):
                backend_status['fallback'] = {
                    'available': True,
                    'type': 'classical'
                }
        except:
            backend_status['fallback'] = {
                'available': False,
                'type': 'unknown'
            }
        
        # Overall health determination
        overall_health = "healthy"
        if self.health_metrics.success_rate < 0.5:
            overall_health = "critical"
        elif self.health_metrics.success_rate < 0.8 or len(recent_errors) > 10:
            overall_health = "degraded"
        elif len(recent_errors) > 5:
            overall_health = "warning"
        
        return {
            'timestamp': current_time,
            'overall_health': overall_health,
            'metrics': {
                'success_rate': self.health_metrics.success_rate,
                'average_solve_time': self.health_metrics.average_solve_time,
                'total_assignments': self.health_metrics.total_assignments,
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'last_success': self.health_metrics.last_success
            },
            'backends': backend_status,
            'configuration': {
                'validation_level': self.validation_level.value,
                'monitoring_enabled': self.enable_monitoring,
                'caching_enabled': self.enable_caching,
                'timeout_seconds': self.timeout_seconds,
                'max_retries': self.max_retries
            },
            'cache_stats': {
                'cached_solutions': len(self.solution_cache),
                'cache_enabled': self.enable_caching
            },
            'recent_error_summary': [
                {
                    'type': error.error_type,
                    'severity': error.severity.value,
                    'message': error.message[:100] + ('...' if len(error.message) > 100 else ''),
                    'timestamp': error.timestamp
                }
                for error in recent_errors[-5:]  # Last 5 recent errors
            ]
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear solution cache and return statistics."""
        cache_size = len(self.solution_cache)
        self.solution_cache.clear()
        
        self.logger.info(f"Cache cleared: {cache_size} solutions removed")
        
        return {
            'cache_cleared': True,
            'solutions_removed': cache_size,
            'timestamp': time.time()
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis and patterns."""
        if not self.error_history:
            return {
                'total_errors': 0,
                'message': 'No errors recorded'
            }
        
        # Error type analysis
        error_types = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Recent error trend (last 24 hours)
        current_time = time.time()
        recent_errors = [
            error for error in self.error_history
            if current_time - error.timestamp < 86400  # 24 hours
        ]
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'severity_distribution': severity_counts,
            'recent_errors_24h': len(recent_errors),
            'error_rate': len(recent_errors) / max(1, self.health_metrics.total_assignments),
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            'latest_errors': [
                {
                    'type': error.error_type,
                    'message': error.message,
                    'severity': error.severity.value,
                    'timestamp': error.timestamp,
                    'suggested_action': error.suggested_action
                }
                for error in self.error_history[-10:]  # Latest 10 errors
            ]
        }


def demo_robust_usage():
    """Demonstrate robust planner capabilities."""
    print("🛡️ Robust Quantum Task Planner Demo")
    print("=" * 60)
    
    # Initialize with different validation levels
    print("\\n🔍 Testing different validation levels...")
    
    for validation_level in [ValidationLevel.BASIC, ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
        print(f"\\n--- {validation_level.value.upper()} Validation Level ---")
        
        planner = RobustQuantumPlanner(
            validation_level=validation_level,
            enable_monitoring=True,
            enable_caching=True,
            max_retries=2
        )
        
        # Test with valid data
        agents = [
            {'id': 'alice', 'skills': ['python', 'ml'], 'capacity': 3},
            {'id': 'bob', 'skills': ['javascript'], 'capacity': 2}
        ]
        
        tasks = [
            {'id': 'backend', 'skills': ['python'], 'priority': 5, 'duration': 2},
            {'id': 'frontend', 'skills': ['javascript'], 'priority': 3, 'duration': 3}
        ]
        
        result = planner.assign_tasks_robust(agents, tasks, minimize="time")
        
        if result['success']:
            print(f"✅ Assignment successful")
            print(f"   Quality Score: {result['quality_analysis']['quality_score']:.2f}")
            print(f"   Assignment Time: {result['metrics']['assignment_time']:.3f}s")
            print(f"   Planner Used: {result['metrics']['planner_used']}")
        else:
            print(f"❌ Assignment failed: {result['error']}")
    
    # Test error handling with invalid data
    print("\\n🚨 Testing error handling with invalid data...")
    
    robust_planner = RobustQuantumPlanner(validation_level=ValidationLevel.STRICT)
    
    # Test missing required fields
    invalid_agents = [
        {'id': 'alice'},  # Missing skills
        {'skills': ['python']}  # Missing id
    ]
    
    invalid_tasks = [
        {'id': 'task1', 'skills': ['python']},
        {'id': 'task2'}  # Missing skills  
    ]
    
    try:
        result = robust_planner.assign_tasks_robust(invalid_agents, invalid_tasks)
        print(f"Validation Error Result: {result['success']} - {result.get('message', 'No message')}")
    except ValueError as e:
        print(f"✅ Validation correctly caught error: {str(e)[:100]}...")
    
    # Test skill mismatch
    print("\\n🎯 Testing skill mismatch detection...")
    
    mismatched_agents = [{'id': 'java_dev', 'skills': ['java'], 'capacity': 1}]
    mismatched_tasks = [{'id': 'python_task', 'skills': ['python'], 'priority': 1, 'duration': 1}]
    
    try:
        result = robust_planner.assign_tasks_robust(mismatched_agents, mismatched_tasks)
        print(f"Skill Mismatch Result: {result['success']} - {result.get('error', 'Success')}")
    except ValueError as e:
        print(f"✅ Skill mismatch correctly detected: Skills not available - python")
    
    # Show health status
    print("\\n📊 Health Status:")
    health = robust_planner.get_health_status()
    print(f"   Overall Health: {health['overall_health']}")
    print(f"   Success Rate: {health['metrics']['success_rate']:.1%}")
    print(f"   Total Errors: {health['metrics']['total_errors']}")
    
    # Show error analysis
    print("\\n🔍 Error Analysis:")
    error_analysis = robust_planner.get_error_analysis()
    print(f"   Total Errors: {error_analysis['total_errors']}")
    if error_analysis.get('most_common_error'):
        print(f"   Most Common Error: {error_analysis['most_common_error']}")


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_robust_usage()