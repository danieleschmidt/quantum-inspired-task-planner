#!/usr/bin/env python3
"""
Production-Ready Quantum Task Planner Deployment
Complete deployment package with all three generations, monitoring, and operational tools.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from simple_quantum_planner import SimpleQuantumPlanner, quick_assign
from robust_quantum_planner import RobustQuantumPlanner, ValidationLevel
from scalable_quantum_planner import ScalableQuantumPlanner, ScalingStrategy, LoadBalancingStrategy


class PlannerType(Enum):
    """Available planner types for production deployment."""
    SIMPLE = "simple"
    ROBUST = "robust"
    SCALABLE = "scalable"


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    planner_type: PlannerType = PlannerType.SCALABLE
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_security: bool = True
    worker_pool_size: Optional[int] = None
    scaling_strategy: ScalingStrategy = ScalingStrategy.BALANCED
    validation_level: ValidationLevel = ValidationLevel.STRICT
    max_requests_per_second: int = 1000
    health_check_interval: int = 30
    log_level: str = "INFO"


class ProductionQuantumPlanner:
    """
    Production-ready quantum task planner with unified interface,
    comprehensive monitoring, and operational excellence.
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize production planner with deployment configuration.
        
        Args:
            config: Deployment configuration (uses defaults if None)
        """
        self.config = config or DeploymentConfig()
        self.start_time = time.time()
        
        # Initialize the appropriate planner based on config
        self._initialize_planner()
        
        # Production metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_history = []
        
        print(f"🚀 ProductionQuantumPlanner initialized")
        print(f"   Type: {self.config.planner_type.value}")
        print(f"   Monitoring: {self.config.enable_monitoring}")
        print(f"   Security: {self.config.enable_security}")
        print(f"   Caching: {self.config.enable_caching}")
        
    def _initialize_planner(self) -> None:
        """Initialize the appropriate planner type."""
        if self.config.planner_type == PlannerType.SIMPLE:
            self.planner = SimpleQuantumPlanner()
            self._assign_method = self._assign_simple
            
        elif self.config.planner_type == PlannerType.ROBUST:
            self.planner = RobustQuantumPlanner(
                validation_level=self.config.validation_level,
                enable_monitoring=self.config.enable_monitoring,
                enable_caching=self.config.enable_caching,
                max_retries=3
            )
            self._assign_method = self._assign_robust
            
        elif self.config.planner_type == PlannerType.SCALABLE:
            self.planner = ScalableQuantumPlanner(
                worker_pool_size=self.config.worker_pool_size,
                scaling_strategy=self.config.scaling_strategy,
                cache_size=10000 if self.config.enable_caching else 0,
                performance_monitoring=self.config.enable_monitoring
            )
            self._assign_method = self._assign_scalable
            
        else:
            raise ValueError(f"Unknown planner type: {self.config.planner_type}")
    
    def assign_tasks(
        self,
        agents: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        minimize: str = "time",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Universal task assignment interface for production use.
        
        Args:
            agents: List of agent dictionaries
            tasks: List of task dictionaries
            minimize: Optimization objective ('time' or 'cost')
            **kwargs: Additional parameters
            
        Returns:
            Standardized result dictionary
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Rate limiting check
            if self._is_rate_limited():
                return self._create_error_result(
                    "Rate limit exceeded",
                    error_type="RateLimitExceeded"
                )
            
            # Security validation if enabled
            if self.config.enable_security:
                security_check = self._validate_security(agents, tasks)
                if not security_check['valid']:
                    return self._create_error_result(
                        f"Security validation failed: {security_check['reason']}",
                        error_type="SecurityValidationError"
                    )
            
            # Execute assignment using the configured planner
            result = self._assign_method(agents, tasks, minimize, **kwargs)
            
            # Standardize result format
            standardized_result = self._standardize_result(result, start_time)
            
            # Update metrics
            if standardized_result['success']:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # Record request for monitoring
            self._record_request(standardized_result, time.time() - start_time)
            
            return standardized_result
            
        except Exception as e:
            self.failed_requests += 1
            error_result = self._create_error_result(str(e), type(e).__name__)
            self._record_request(error_result, time.time() - start_time)
            return error_result
    
    def _assign_simple(self, agents: List[Dict], tasks: List[Dict], minimize: str, **kwargs) -> Dict:
        """Execute assignment with simple planner."""
        return self.planner.assign_tasks(agents, tasks, minimize)
    
    def _assign_robust(self, agents: List[Dict], tasks: List[Dict], minimize: str, **kwargs) -> Dict:
        """Execute assignment with robust planner."""
        return self.planner.assign_tasks_robust(agents, tasks, minimize)
    
    def _assign_scalable(self, agents: List[Dict], tasks: List[Dict], minimize: str, **kwargs) -> Dict:
        """Execute assignment with scalable planner."""
        return self.planner.assign_tasks_scalable(agents, tasks, minimize)
    
    def _is_rate_limited(self) -> bool:
        """Check if request should be rate limited."""
        current_time = time.time()
        
        # Remove old requests (older than 1 second)
        recent_requests = [
            req_time for req_time in self.request_history
            if current_time - req_time < 1.0
        ]
        
        self.request_history = recent_requests
        
        return len(recent_requests) >= self.config.max_requests_per_second
    
    def _validate_security(self, agents: List[Dict], tasks: List[Dict]) -> Dict[str, Any]:
        """Perform security validation on input data."""
        # Check for potentially malicious patterns
        dangerous_patterns = ['<script', 'javascript:', 'vbscript:', 'DROP TABLE', 'SELECT *']
        
        def check_string_safety(value: str) -> bool:
            if not isinstance(value, str):
                return True
            value_lower = value.lower()
            return not any(pattern.lower() in value_lower for pattern in dangerous_patterns)
        
        # Validate agents
        for agent in agents:
            if not check_string_safety(agent.get('id', '')):
                return {'valid': False, 'reason': f"Malicious pattern in agent ID: {agent.get('id', '')[:50]}"}
            
            for skill in agent.get('skills', []):
                if not check_string_safety(skill):
                    return {'valid': False, 'reason': f"Malicious pattern in agent skill: {skill[:50]}"}
        
        # Validate tasks
        for task in tasks:
            if not check_string_safety(task.get('id', '')):
                return {'valid': False, 'reason': f"Malicious pattern in task ID: {task.get('id', '')[:50]}"}
            
            for skill in task.get('skills', []):
                if not check_string_safety(skill):
                    return {'valid': False, 'reason': f"Malicious pattern in task skill: {skill[:50]}"}
        
        # Check for reasonable data sizes
        if len(agents) > 10000 or len(tasks) > 10000:
            return {'valid': False, 'reason': 'Input data too large (DoS protection)'}
        
        return {'valid': True, 'reason': None}
    
    def _standardize_result(self, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Standardize result format across all planner types."""
        return {
            # Core fields (guaranteed to exist)
            'success': result.get('success', False),
            'assignments': result.get('assignments', {}),
            'completion_time': result.get('completion_time', 0),
            'total_cost': result.get('total_cost', result.get('cost', 0)),
            'backend_used': result.get('backend_used', 'unknown'),
            'message': result.get('message', 'Assignment completed'),
            
            # Metadata
            'metadata': {
                'planner_type': self.config.planner_type.value,
                'execution_time': time.time() - start_time,
                'timestamp': time.time(),
                'request_id': self.total_requests,
                'cache_hit': result.get('cache_hit', False)
            },
            
            # Performance metrics (if available)
            'performance': result.get('performance', result.get('metrics', {})),
            
            # Quality metrics (if available) 
            'quality': result.get('quality', result.get('quality_analysis', {})),
            
            # Error information (if applicable)
            'error': result.get('error'),
            'error_type': result.get('error_type'),
            'diagnostics': result.get('diagnostics', {})
        }
    
    def _create_error_result(self, error_message: str, error_type: str = "UnknownError") -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'success': False,
            'assignments': {},
            'completion_time': 0,
            'total_cost': 0,
            'backend_used': 'none',
            'message': f'Assignment failed: {error_message}',
            'error': error_message,
            'error_type': error_type,
            'metadata': {
                'planner_type': self.config.planner_type.value,
                'execution_time': 0.0,
                'timestamp': time.time(),
                'request_id': self.total_requests
            }
        }
    
    def _record_request(self, result: Dict[str, Any], execution_time: float) -> None:
        """Record request for monitoring and metrics."""
        self.request_history.append(time.time())
        
        # Keep only recent history (last 1000 requests)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for production monitoring."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate success rate
        success_rate = self.successful_requests / max(self.total_requests, 1)
        
        # Calculate requests per second (last minute)
        recent_requests = [
            req_time for req_time in self.request_history
            if current_time - req_time < 60
        ]
        requests_per_second = len(recent_requests) / 60.0
        
        # Get planner-specific health if available
        planner_health = {}
        if hasattr(self.planner, 'get_health_status'):
            planner_health = self.planner.get_health_status()
        elif hasattr(self.planner, 'get_comprehensive_status'):
            planner_health = self.planner.get_comprehensive_status()
        
        # Determine overall health
        overall_health = "healthy"
        if success_rate < 0.5:
            overall_health = "critical"
        elif success_rate < 0.8 or requests_per_second > self.config.max_requests_per_second * 0.9:
            overall_health = "warning"
        
        return {
            'timestamp': current_time,
            'uptime_seconds': uptime,
            'overall_health': overall_health,
            'planner_type': self.config.planner_type.value,
            'metrics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': success_rate,
                'requests_per_second': requests_per_second
            },
            'configuration': {
                'monitoring_enabled': self.config.enable_monitoring,
                'caching_enabled': self.config.enable_caching,
                'security_enabled': self.config.enable_security,
                'max_rps': self.config.max_requests_per_second
            },
            'planner_health': planner_health
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        # Get planner-specific performance stats
        planner_stats = {}
        if hasattr(self.planner, 'get_performance_stats'):
            planner_stats = self.planner.get_performance_stats()
        
        return {
            'timestamp': time.time(),
            'production_stats': {
                'total_requests': self.total_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'average_rps': len(self.request_history) / max(time.time() - self.start_time, 1)
            },
            'planner_stats': planner_stats,
            'configuration': {
                'planner_type': self.config.planner_type.value,
                'worker_pool_size': getattr(self.config, 'worker_pool_size', None),
                'scaling_strategy': getattr(self.config, 'scaling_strategy', None)
            }
        }
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check for load balancer/monitoring."""
        try:
            # Test basic functionality
            test_agents = [{'id': 'test_agent', 'skills': ['test'], 'capacity': 1}]
            test_tasks = [{'id': 'test_task', 'skills': ['test'], 'priority': 1, 'duration': 1}]
            
            start_time = time.time()
            test_result = self.assign_tasks(test_agents, test_tasks, minimize="time")
            response_time = time.time() - start_time
            
            # Health check result
            is_healthy = (
                test_result['success'] and
                response_time < 5.0 and  # Should respond within 5 seconds
                self.successful_requests / max(self.total_requests, 1) > 0.1  # At least 10% success rate
            )
            
            return {
                'healthy': is_healthy,
                'response_time': response_time,
                'test_assignment_success': test_result['success'],
                'overall_success_rate': self.successful_requests / max(self.total_requests, 1),
                'timestamp': time.time(),
                'details': {
                    'planner_type': self.config.planner_type.value,
                    'uptime': time.time() - self.start_time,
                    'total_requests': self.total_requests
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'response_time': 0,
                'test_assignment_success': False,
                'timestamp': time.time()
            }


def create_production_planner(
    planner_type: str = "scalable",
    monitoring: bool = True,
    caching: bool = True,
    security: bool = True
) -> ProductionQuantumPlanner:
    """
    Factory function to create production-ready quantum task planner.
    
    Args:
        planner_type: "simple", "robust", or "scalable"
        monitoring: Enable monitoring and metrics
        caching: Enable result caching
        security: Enable security validation
        
    Returns:
        Configured ProductionQuantumPlanner instance
    """
    config = DeploymentConfig(
        planner_type=PlannerType(planner_type),
        enable_monitoring=monitoring,
        enable_caching=caching,
        enable_security=security
    )
    
    return ProductionQuantumPlanner(config)


def demo_production_deployment():
    """Demonstrate production deployment capabilities."""
    print("🚀 PRODUCTION QUANTUM TASK PLANNER DEPLOYMENT DEMO")
    print("=" * 80)
    
    # Test all three planner types in production configuration
    planner_configs = [
        {"type": "simple", "name": "Simple Production"},
        {"type": "robust", "name": "Robust Production"}, 
        {"type": "scalable", "name": "Scalable Production"}
    ]
    
    for config in planner_configs:
        print(f"\\n🔧 Testing {config['name']} Configuration")
        print("-" * 50)
        
        planner = create_production_planner(
            planner_type=config['type'],
            monitoring=True,
            caching=True,
            security=True
        )
        
        # Test data
        agents = [
            {'id': 'prod_agent_1', 'skills': ['python', 'ml'], 'capacity': 3},
            {'id': 'prod_agent_2', 'skills': ['javascript', 'react'], 'capacity': 2},
            {'id': 'prod_agent_3', 'skills': ['python', 'devops'], 'capacity': 2}
        ]
        
        tasks = [
            {'id': 'prod_backend', 'skills': ['python'], 'priority': 5, 'duration': 2},
            {'id': 'prod_frontend', 'skills': ['javascript', 'react'], 'priority': 3, 'duration': 3},
            {'id': 'prod_ml', 'skills': ['python', 'ml'], 'priority': 8, 'duration': 4},
            {'id': 'prod_deploy', 'skills': ['devops'], 'priority': 6, 'duration': 1}
        ]
        
        # Test normal operation
        start_time = time.time()
        result = planner.assign_tasks(agents, tasks, minimize="time")
        execution_time = time.time() - start_time
        
        print(f"✅ Assignment: {result['success']}")
        print(f"   Time: {execution_time:.3f}s")
        print(f"   Backend: {result['backend_used']}")
        print(f"   Assignments: {len(result['assignments'])}")
        
        # Test health check
        health = planner.perform_health_check()
        print(f"   Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
        
        # Test security (malicious input)
        print("\\n🔒 Security Test:")
        malicious_agents = [{'id': '<script>alert("xss")</script>', 'skills': ['python'], 'capacity': 1}]
        malicious_result = planner.assign_tasks(malicious_agents, tasks[:1])
        print(f"   Malicious input blocked: {'✅' if not malicious_result['success'] else '❌'}")
        
        # Performance test
        print("\\n⚡ Performance Test:")
        performance_times = []
        for i in range(5):
            start = time.time()
            test_result = planner.assign_tasks(agents[:2], tasks[:2], minimize="time")
            if test_result['success']:
                performance_times.append(time.time() - start)
        
        if performance_times:
            avg_time = sum(performance_times) / len(performance_times)
            min_time = min(performance_times)
            max_time = max(performance_times)
            print(f"   5 requests - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
        
        # Get final status
        status = planner.get_health_status()
        print(f"\\n📊 Final Status:")
        print(f"   Requests: {status['metrics']['total_requests']}")
        print(f"   Success Rate: {status['metrics']['success_rate']:.1%}")
        print(f"   RPS: {status['metrics']['requests_per_second']:.1f}")
        print(f"   Overall Health: {status['overall_health']}")
    
    print("\\n🎯 Production deployment demonstration complete!")
    print("\\n💡 Ready for production deployment with:")
    print("   • Three planner types (Simple, Robust, Scalable)")
    print("   • Comprehensive monitoring and health checks")
    print("   • Security validation and rate limiting") 
    print("   • Standardized API interface")
    print("   • Production-grade error handling")


if __name__ == "__main__":
    demo_production_deployment()